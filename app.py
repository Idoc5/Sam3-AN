import os
import sys
import subprocess
import threading
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# 添加SAM3到路径 (使用本地 SAM_src 目录)
sam3_src = Path(__file__).parent / "SAM_src"
sys.path.insert(0, str(sam3_src))

from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import json
import uuid
import requests
from datetime import datetime

from services.sam3_service import SAM3Service
from services.annotation_manager import AnnotationManager
from services.sam3_concurrent_service import SAM3ConcurrentService
from services.db_annotation_manager import DBAnnotationManager
from exports.yolo_exporter import YOLOExporter
from exports.coco_exporter import COCOExporter
from config.segmentation_config import SegmentationConfig

app = Flask(__name__)
CORS(app)

# 全局配置
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB
app.config['UPLOAD_FOLDER'] = Path(__file__).parent / 'uploads'
app.config['UPLOAD_FOLDER'].mkdir(exist_ok=True)

# 全局服务实例
sam3_service = None
sam3_concurrent = None  # 并发服务
annotation_manager = None  # JSON管理器（仅在USE_SQLITE=False时使用）
db_manager = DBAnnotationManager()  # SQLite管理器（默认使用）

# 批量处理进度状态
batch_progress = {}  # {task_id: {'current': 0, 'total': 0, 'status': 'processing/done/error'}}
batch_progress_lock = threading.Lock()

# 后台任务执行器（用于异步处理批量任务）
background_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="batch_task_")


# 允许的图片后缀名
ALLOWED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.tif'}

# 允许的视频后缀名
ALLOWED_VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webp', '.flv', '.wmv'}


def strip_image_paths(images: list) -> list:
    """移除图片列表中的路径信息（安全考虑，不暴露给前端）
    
    Args:
        images: 图片列表
        
    Returns:
        不包含path字段的图片列表
    """
    return [
        {k: v for k, v in img.items() if k != 'path'}
        for img in images
    ]


def validate_video_path(video_path: str) -> tuple[bool, str]:
    """验证视频路径是否安全
    
    Args:
        video_path: 视频路径
        
    Returns:
        (是否安全, 错误信息)
    """
    if not video_path:
        return False, '缺少视频路径'
    
    # 安全检查：禁止路径遍历
    if '..' in video_path:
        return False, '非法路径：禁止路径遍历'
    
    # 安全检查：只允许常见视频后缀名
    path_lower = video_path.lower()
    if not any(path_lower.endswith(ext) for ext in ALLOWED_VIDEO_EXTENSIONS):
        return False, '不支持的视频文件类型'
    
    # 检查文件是否存在
    if not os.path.exists(video_path):
        return False, '视频文件不存在'
    
    return True, ''


def get_sam3_service():
    """延迟加载SAM3服务"""
    global sam3_service
    if sam3_service is None:
        sam3_service = SAM3Service()
    return sam3_service


def get_sam3_concurrent():
    """延迟加载SAM3并发服务"""
    global sam3_concurrent
    if sam3_concurrent is None:
        sam3_concurrent = SAM3ConcurrentService(
            num_workers=SegmentationConfig.NUM_WORKERS,
            batch_size=SegmentationConfig.BATCH_SIZE,
            cache_size=SegmentationConfig.CACHE_SIZE,
            max_image_size=SegmentationConfig.MAX_IMAGE_SIZE
        )
    return sam3_concurrent


def reset_sam3_service():
    """重置SAM3服务以应用新配置"""
    global sam3_service
    if sam3_service is not None:
        # 清理旧服务
        if hasattr(sam3_service, 'cleanup'):
            try:
                sam3_service.cleanup()
            except:
                pass
        sam3_service = None
    return get_sam3_service()


def reset_sam3_concurrent():
    """重置SAM3并发服务以应用新配置"""
    global sam3_concurrent
    if sam3_concurrent is not None:
        # 清理旧服务
        if hasattr(sam3_concurrent, 'cleanup'):
            try:
                sam3_concurrent.cleanup()
            except:
                pass
        sam3_concurrent = None
    return get_sam3_concurrent()


def get_db_manager():
    """获取数据库管理器（默认使用SQLite）"""
    if SegmentationConfig.USE_SQLITE:
        return db_manager
    else:
        # 仅当明确设置USE_SQLITE=False时才使用JSON模式
        if annotation_manager is None:
            annotation_manager = AnnotationManager()
        return annotation_manager


# ==================== 页面路由 ====================

@app.route('/')
def index():
    """主页面"""
    return render_template('index.html')


@app.route('/video')
def video_page():
    """视频标注页面"""
    return render_template('video.html')


# ==================== 项目管理API ====================

@app.route('/api/project/create', methods=['POST'])
def create_project():
    """创建新项目"""
    data = request.json
    project_id = str(uuid.uuid4())[:8]
    project = {
        'id': project_id,
        'name': data.get('name', f'项目_{project_id}'),
        'image_dir': data.get('image_dir', ''),
        'output_dir': data.get('output_dir', ''),
        'export_format': data.get('export_format', 'yolo'),
        'classes': data.get('classes', []),
        'created_at': datetime.now().isoformat(),
        'images': [],
        'current_index': 0
    }
    manager = get_db_manager()
    manager.create_project(project)
    return jsonify({'success': True, 'project': project})


@app.route('/api/project/<project_id>', methods=['GET'])
def get_project(project_id):
    """获取项目信息"""
    manager = get_db_manager()
    project = manager.get_project(project_id)
    if project:
        # 移除图片路径，不暴露给前端
        if 'images' in project:
            project['images'] = strip_image_paths(project['images'])
        return jsonify({'success': True, 'project': project})
    return jsonify({'success': False, 'error': '项目不存在'})


@app.route('/api/project/<project_id>/info', methods=['GET'])
def get_project_info(project_id):
    """获取项目基本信息（不包含图片列表和标注，轻量级）"""
    manager = get_db_manager()
    project = manager.get_project(project_id)
    if project:
        # 只返回基本信息，不返回 images 数组（避免数据过大）
        return jsonify({
            'success': True,
            'project': {
                'id': project.get('id'),
                'name': project.get('name'),
                'image_dir': project.get('image_dir'),
                'output_dir': project.get('output_dir'),
                'export_format': project.get('export_format'),
                'classes': project.get('classes', []),
                'image_count': len(project.get('images', [])),
                'annotated_count': sum(1 for img in project.get('images', []) if img.get('annotated', False))
            }
        })
    return jsonify({'success': False, 'error': '项目不存在'})


@app.route('/api/project/<project_id>/images', methods=['GET'])
def get_project_images(project_id):
    """获取项目图片列表（仅图片基本信息，不含标注数据）"""
    manager = get_db_manager()
    project = manager.get_project(project_id)
    if project:
        # 返回图片的基本信息（不暴露实际路径）
        images = [
            {
                'id': img.get('id'),
                'filename': img.get('filename'),
                'annotated': img.get('annotated', False)
            }
            for img in project.get('images', [])
        ]
        return jsonify({
            'success': True,
            'images': images,
            'total': len(images)
        })
    return jsonify({'success': False, 'error': '项目不存在'})


@app.route('/api/project/<project_id>/update', methods=['POST'])
def update_project(project_id):
    """更新项目信息"""
    data = request.json
    manager = get_db_manager()
    project = manager.get_project(project_id)

    if not project:
        return jsonify({'success': False, 'error': '项目不存在'})

    # 构建更新字段
    updates = {}
    if 'name' in data:
        updates['name'] = data['name']
    if 'image_dir' in data:
        updates['image_dir'] = data['image_dir']
    if 'output_dir' in data:
        updates['output_dir'] = data['output_dir']
    if 'classes' in data:
        updates['classes'] = data['classes']

    try:
        updated_project = manager.update_project(project_id, updates)
        return jsonify({'success': True, 'project': updated_project})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/project/<project_id>/delete', methods=['POST'])
def delete_project(project_id):
    """删除项目"""
    manager = get_db_manager()
    project = manager.get_project(project_id)

    if not project:
        return jsonify({'success': False, 'error': '项目不存在'})

    try:
        manager.delete_project(project_id)
        return jsonify({'success': True, 'message': '项目已删除'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/project/<project_id>/load_images', methods=['POST'])
def load_project_images(project_id):
    """加载项目图片目录"""
    data = request.json
    image_dir = data.get('image_dir', '')

    if not os.path.isdir(image_dir):
        return jsonify({'success': False, 'error': '目录不存在'})

    # 支持的图片格式
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}
    images = []

    for f in sorted(os.listdir(image_dir)):
        if Path(f).suffix.lower() in extensions:
            images.append({
                'filename': f,
                'path': os.path.join(image_dir, f),  # 内部使用，不返回给前端
                'annotated': False,
                'annotations': []
            })

    manager = get_db_manager()
    manager.update_project_images(project_id, images, image_dir)

    # 从数据库重新获取图片列表（不暴露实际路径）
    project = manager.get_project(project_id)
    if project:
        images = [
            {
                'id': img.get('id'),
                'filename': img.get('filename'),
                'annotated': img.get('annotated', False)
            }
            for img in project.get('images', [])
        ]

    return jsonify({
        'success': True,
        'count': len(images),
        'images': images
    })


@app.route('/api/project/list', methods=['GET'])
def list_projects():
    """列出所有项目（轻量级，不含详细标注数据）"""
    manager = get_db_manager()
    projects = manager.list_projects()
    # 数据库已经返回了统计信息，直接使用
    return jsonify({'success': True, 'projects': projects})


# ==================== 图片服务API ====================

@app.route('/api/image/serve')
def serve_image():
    """提供图片文件（通过image_id获取）"""
    image_id = request.args.get('image_id', '')
    if not image_id:
        return jsonify({'error': '缺少图片ID'}), 400

    # 验证 image_id 为整数
    try:
        image_id = int(image_id)
    except ValueError:
        return jsonify({'error': '无效的图片ID'}), 400

    # 从数据库获取图片路径
    manager = get_db_manager()
    image = manager.get_image(image_id)
    if not image:
        return jsonify({'error': '图片不存在'}), 404

    path = image.get('path')
    if not path:
        return jsonify({'error': '图片路径无效'}), 404

    # 安全检查：只允许常见图片后缀名
    path_lower = path.lower()
    if not any(path_lower.endswith(ext) for ext in ALLOWED_IMAGE_EXTENSIONS):
        return jsonify({'error': '不支持的文件类型'}), 400

    abs_path = Path(path).resolve()
    if not abs_path.is_file():
        return jsonify({'error': '文件不存在'}), 404

    return send_file(abs_path)


# ==================== SAM3分割API ====================

@app.route('/api/segment/text', methods=['POST'])
def segment_by_text():
    """文本提示分割（单线程串行处理）"""
    print("\n[MODE] 当前模式: 单线程串行处理 (/api/segment/text)")
    print("[INFO] 如需使用并发优化，请调用 /api/segment/batch/concurrent 接口\n")

    data = request.json
    image_id = data.get('image_id')
    prompt = data.get('prompt', '')
    confidence = data.get('confidence', 0.5)

    if not image_id or not prompt:
        return jsonify({'success': False, 'error': '缺少必要参数'})

    # 从数据库获取图片路径
    manager = get_db_manager()
    image = manager.get_image(image_id)
    if not image:
        return jsonify({'success': False, 'error': '图片不存在'})
    
    image_path = image.get('path')
    if not image_path:
        return jsonify({'success': False, 'error': '图片路径无效'})

    try:
        service = get_sam3_service()
        results = service.segment_by_text(image_path, prompt, confidence)
        if results:
            print(f"[INFO] 文本分割完成: {len(results)} 个结果")
        return jsonify({'success': True, 'results': results})
    except Exception as e:
        print(f"[ERROR] segment_by_text 异常: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/segment/point', methods=['POST'])
def segment_by_point():
    """点击分割"""
    data = request.json
    image_id = data.get('image_id')
    points = data.get('points', [])  # [[x, y, label], ...]

    if not image_id or not points:
        return jsonify({'success': False, 'error': '缺少必要参数'})

    # 从数据库获取图片路径
    manager = get_db_manager()
    image = manager.get_image(image_id)
    if not image:
        return jsonify({'success': False, 'error': '图片不存在'})
    
    image_path = image.get('path')
    if not image_path:
        return jsonify({'success': False, 'error': '图片路径无效'})

    try:
        service = get_sam3_service()
        results = service.segment_by_points(image_path, points)
        return jsonify({'success': True, 'results': results})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/segment/box', methods=['POST'])
def segment_by_box():
    """框选分割"""
    data = request.json
    image_id = data.get('image_id')
    boxes = data.get('boxes', [])  # [[x1, y1, x2, y2, label], ...]

    if not image_id or not boxes:
        return jsonify({'success': False, 'error': '缺少必要参数'})

    # 从数据库获取图片路径
    manager = get_db_manager()
    image = manager.get_image(image_id)
    if not image:
        return jsonify({'success': False, 'error': '图片不存在'})
    
    image_path = image.get('path')
    if not image_path:
        return jsonify({'success': False, 'error': '图片路径无效'})

    try:
        service = get_sam3_service()
        results = service.segment_by_boxes(image_path, boxes)
        return jsonify({'success': True, 'results': results})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/segment/batch', methods=['POST'])
def batch_segment():
    """批量分割"""
    data = request.json
    project_id = data.get('project_id')
    prompt = data.get('prompt', '')
    class_name = data.get('class_name', prompt)  # 使用传入的类名，默认为prompt
    start_index = data.get('start_index', 0)
    end_index = data.get('end_index', -1)
    skip_annotated = data.get('skip_annotated', True)
    confidence = data.get('confidence', 0.5)

    try:
        service = get_sam3_service()
        manager = get_db_manager()
        project = manager.get_project(project_id)

        if not project:
            return jsonify({'success': False, 'error': '项目不存在'})

        images = project['images']
        if end_index == -1:
            end_index = len(images)

        processed = 0
        failed = 0
        total_detections = 0
        results = []

        for i in range(start_index, min(end_index, len(images))):
            img_info = images[i]

            # 跳过已标注的图片
            if skip_annotated and img_info.get('annotated', False):
                continue

            try:
                seg_results = service.segment_by_text(
                    img_info['path'], prompt, confidence
                )

                if seg_results:
                    # 使用传入的类名，而不是 prompt
                    for r in seg_results:
                        r['class_name'] = class_name

                    manager.add_annotations(
                        project_id, i, seg_results, class_name
                    )
                    processed += 1
                    total_detections += len(seg_results)
                    results.append({
                        'index': i,
                        'filename': img_info['filename'],
                        'count': len(seg_results)
                    })
                else:
                    # 没有检测到对象，也算处理过
                    processed += 1

            except Exception as e:
                print(f"[ERROR] 批量分割图片 {img_info['filename']} 失败: {e}")
                failed += 1
                continue

        return jsonify({
            'success': True,
            'processed': processed,
            'failed': failed,
            'total_detections': total_detections,
            'results': results
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        # 更新进度为错误
        with batch_progress_lock:
            if task_id in batch_progress:
                batch_progress[task_id]['status'] = 'error'
                batch_progress[task_id]['message'] = str(e)
        return jsonify({'success': False, 'error': str(e), 'task_id': task_id})


@app.route('/api/segment/batch/progress', methods=['GET'])
def batch_segment_progress():
    """查询批量处理进度"""
    import time
    task_id = request.args.get('task_id')

    if not task_id:
        return jsonify({'success': False, 'error': '缺少task_id'})

    with batch_progress_lock:
        progress_info = batch_progress.get(task_id)

    if not progress_info:
        return jsonify({'success': False, 'error': '任务不存在'})

    # 清理已完成的旧任务（5分钟后）
    current_time = time.time()
    with batch_progress_lock:
        if progress_info['status'] in ['done', 'error']:
            # 保留一段时间供前端查询
            if current_time - progress_info.get('completed_at', 0) > 300:
                del batch_progress[task_id]

    return jsonify({
        'success': True,
        'progress': progress_info
    })


@app.route('/api/segment/batch/concurrent', methods=['POST'])
def batch_segment_concurrent():
    """并发批量分割（异步优化版）- 后台执行，防止504超时"""
    import uuid

    print("\n" + "=" * 60)
    print("[MODE] 当前模式: 并发批量处理 (异步后台) (/api/segment/batch/concurrent)")
    print("=" * 60 + "\n")

    data = request.json
    project_id = data.get('project_id')
    prompt = data.get('prompt', '')
    class_name = data.get('class_name', prompt)
    start_index = data.get('start_index', 0)
    end_index = data.get('end_index', -1)
    skip_annotated = data.get('skip_annotated', True)
    confidence = data.get('confidence', 0.5)
    use_cache = data.get('use_cache', True)
    max_workers = data.get('max_workers', 2)

    # 生成任务ID
    task_id = data.get('task_id', str(uuid.uuid4()))

    try:
        # 获取项目
        manager = get_db_manager()
        project = manager.get_project(project_id)

        if not project:
            with batch_progress_lock:
                batch_progress[task_id] = {
                    'current': 0,
                    'total': 0,
                    'status': 'error',
                    'message': '项目不存在'
                }
            return jsonify({'success': False, 'error': '项目不存在', 'task_id': task_id})

        images = project['images']
        if end_index == -1:
            end_index = len(images)

        # 收集待处理图片
        to_process = []
        skipped_count = 0
        for i in range(start_index, min(end_index, len(images))):
            img_info = images[i]
            is_annotated = img_info.get('annotated', False)
            if skip_annotated and is_annotated:
                skipped_count += 1
                continue
            to_process.append((i, img_info['path'], img_info['filename']))

        total_to_process = len(to_process)

        print(f"[API] 任务 {task_id}: 原始范围 [{start_index}, {min(end_index, len(images))}), 总数={min(end_index, len(images)) - start_index}")
        print(f"[API] 任务 {task_id}: skip_annotated={skip_annotated}, 跳过已标注={skipped_count}")
        print(f"[API] 任务 {task_id}: 待处理图片 {total_to_process} 张")
        if total_to_process <= 10:
            print(f"[API] 待处理列表: {[(idx, fname) for idx, _, fname in to_process]}")

        # 初始化进度
        with batch_progress_lock:
            batch_progress[task_id] = {
                'current': 0,
                'total': total_to_process,
                'status': 'processing'
            }

        # 提交后台任务
        background_executor.submit(
            _process_batch_in_background,
            task_id,
            project_id,
            images,
            to_process,
            prompt,
            class_name,
            confidence,
            use_cache,
            max_workers
        )

        # 立即返回，不等待处理完成
        return jsonify({
            'success': True,
            'task_id': task_id,
            'message': '批量处理已启动，请在后台继续进行'
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        with batch_progress_lock:
            batch_progress[task_id] = {
                'current': 0,
                'total': 0,
                'status': 'error',
                'message': str(e)
            }
        return jsonify({'success': False, 'error': str(e), 'task_id': task_id})


def _process_batch_in_background(task_id, project_id, images, to_process, prompt,
                               class_name, confidence, use_cache, max_workers):
    """后台执行批量处理（实时保存）"""
    import time
    from utils.performance_monitor import PerformanceMonitor

    perf_monitor = PerformanceMonitor("ConcurrentBatchSegment")

    # 构建路径到索引的映射
    path_to_index = {path: (index, filename) for index, path, filename in to_process}

    # 实时保存回调函数
    processed_counter = [0]  # 使用列表以便在闭包中修改
    total_detections_counter = [0]
    results_list = []

    def on_result_ready(image_path, detections):
        """当单张图片处理完成时立即保存"""
        if image_path in path_to_index:
            index, filename = path_to_index[image_path]

            manager = get_db_manager()

            if detections:
                for det in detections:
                    det['class_name'] = class_name

                # 立即保存到数据库
                save_data = [(project_id, index, detections, class_name)]
                manager.add_annotations_batch(save_data)

                processed_counter[0] += 1
                total_detections_counter[0] += len(detections)
                results_list.append({
                    'index': index,
                    'filename': filename,
                    'count': len(detections)
                })
                print(f"[后台任务] 实时保存: 索引{index}, {filename}, 检测数={len(detections)}, 已处理{processed_counter[0]}/{len(to_process)}")
            else:
                processed_counter[0] += 1
                print(f"[后台任务] 无检测结果: 索引{index}, {filename}, 已处理{processed_counter[0]}/{len(to_process)}")

            # 🔧 实时更新进度（关键修复）
            with batch_progress_lock:
                if task_id in batch_progress:
                    batch_progress[task_id]['current'] = processed_counter[0]
                    batch_progress[task_id]['total_detections'] = total_detections_counter[0]

    def progress_callback(current, total):
        """进度回调（供并发服务调用）"""
        # 注意：这里的 current 是 segment_batch_concurrent 内部的计数
        # 实际进度由 on_result_ready 更新
        pass

    try:
        perf_monitor.marker('inference')
        image_paths = [p[1] for p in to_process]

        # 并发推理（传递实时保存回调）
        concurrent_service = get_sam3_concurrent()
        segment_results = concurrent_service.segment_batch_concurrent(
            image_paths, prompt, confidence, use_cache, max_workers=max_workers,
            on_result_callback=on_result_ready
        )

        perf_monitor.record('inference', perf_monitor.elapsed_since('inference'))

        total_time = perf_monitor.elapsed_since('inference')
        avg_time = total_time / len(to_process) if to_process else 0

        print(f"[后台任务] 批量分割完成!")
        print(f"[后台任务] 总耗时: {total_time:.2f}秒, 平均: {avg_time:.3f}秒/张")
        print(f"[后台任务] 处理: {processed_counter[0]}, 检测: {total_detections_counter[0]}")

        # 更新进度为完成
        with batch_progress_lock:
            if task_id in batch_progress:
                batch_progress[task_id]['status'] = 'done'
                batch_progress[task_id]['current'] = len(to_process)
                batch_progress[task_id]['processed'] = processed_counter[0]
                batch_progress[task_id]['total_detections'] = total_detections_counter[0]
                batch_progress[task_id]['total_time'] = round(total_time, 2)
                batch_progress[task_id]['avg_time_per_image'] = round(avg_time, 3)
                batch_progress[task_id]['completed_at'] = time.time()

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[后台任务] 执行失败: {e}")
        with batch_progress_lock:
            if task_id in batch_progress:
                batch_progress[task_id]['status'] = 'error'
                batch_progress[task_id]['message'] = str(e)
                batch_progress[task_id]['processed'] = processed_counter[0]
                batch_progress[task_id]['total_detections'] = total_detections_counter[0]
                batch_progress[task_id]['completed_at'] = time.time()


# ==================== 标注管理API ====================

@app.route('/api/annotation/save', methods=['POST'])
def save_annotation():
    """保存标注"""
    data = request.json
    project_id = data.get('project_id')
    image_index = data.get('image_index')
    annotations = data.get('annotations', [])

    try:
        manager = get_db_manager()
        manager.save_annotations(project_id, image_index, annotations)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/annotation/add', methods=['POST'])
def add_annotation():
    """增量添加标注（不删除已有标注）"""
    data = request.json
    project_id = data.get('project_id')
    image_index = data.get('image_index')
    annotations = data.get('annotations', [])
    class_name = data.get('class_name', None)

    try:
        manager = get_db_manager()
        manager.add_annotations(project_id, image_index, annotations, class_name)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/annotation/get', methods=['GET'])
def get_annotation():
    """获取标注"""
    project_id = request.args.get('project_id')
    image_id = request.args.get('image_id')  # 优先使用图片ID
    image_index = request.args.get('image_index')  # 兼容旧的索引方式

    manager = get_db_manager()

    if image_id:
        # 使用图片ID查询标注
        annotations = manager.get_annotations_by_image_id(project_id, int(image_id))
    elif image_index is not None:
        # 使用图片索引查询标注（兼容旧方式）
        annotations = manager.get_annotations(project_id, int(image_index))
    else:
        annotations = []

    return jsonify({'success': True, 'annotations': annotations})


@app.route('/api/annotation/update', methods=['POST'])
def update_annotation():
    """更新单个标注（手动调整）"""
    data = request.json
    project_id = data.get('project_id')
    image_index = data.get('image_index')
    annotation_id = data.get('annotation_id')
    updates = data.get('updates', {})

    try:
        manager = get_db_manager()
        manager.update_annotation(
            project_id, image_index, annotation_id, updates
        )
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/annotation/delete', methods=['POST'])
def delete_annotation():
    """删除标注"""
    data = request.json
    project_id = data.get('project_id')
    image_index = data.get('image_index')
    annotation_id = data.get('annotation_id')

    try:
        manager = get_db_manager()
        manager.delete_annotation(project_id, image_index, annotation_id)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/annotation/delete_class', methods=['POST'])
def delete_class_annotations():
    """删除项目中某个类别的所有标注"""
    data = request.json
    project_id = data.get('project_id')
    class_name = data.get('class_name')

    if not project_id or not class_name:
        return jsonify({'success': False, 'error': '缺少项目ID或类别名称'})

    try:
        manager = get_db_manager()
        deleted_count = manager.delete_class_annotations(project_id, class_name)
        return jsonify({
            'success': True,
            'deleted_count': deleted_count,
            'message': f'已删除 {deleted_count} 个 "{class_name}" 类型的标注'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/image/delete', methods=['POST'])
def delete_image():
    """删除图片及其所有标注"""
    data = request.json
    project_id = data.get('project_id')
    image_index = data.get('image_index')

    if not project_id or image_index is None:
        return jsonify({'success': False, 'error': '缺少项目ID或图片索引'})

    try:
        manager = get_db_manager()
        success = manager.delete_image(project_id, image_index)
        if success:
            return jsonify({'success': True, 'message': '图片已删除'})
        else:
            return jsonify({'success': False, 'error': '图片不存在'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/annotation/clear_non_manual', methods=['POST'])
def clear_non_manual_annotations():
    """清理项目中所有非手动标注（AI自动生成的标注）"""
    data = request.json
    project_id = data.get('project_id')

    if not project_id:
        return jsonify({'success': False, 'error': '缺少项目ID'})

    try:
        manager = get_db_manager()
        deleted_count = manager.clear_non_manual_annotations(project_id)
        return jsonify({
            'success': True,
            'deleted_count': deleted_count,
            'message': f'已清理 {deleted_count} 个非手动标注（AI自动生成的标注已删除，手动标注保留）'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/annotation/by_class', methods=['POST'])
def get_annotations_by_class():
    """获取指定类别的标注概况信息"""
    data = request.json
    project_id = data.get('project_id')
    class_name = data.get('class_name')
    manual_only = data.get('manual_only', False)

    if not project_id or not class_name:
        return jsonify({'success': False, 'error': '缺少项目ID或类别名称'})

    try:
        manager = get_db_manager()
        # 默认只返回概况信息，减少数据传输
        summary = manager.get_annotations_by_class(project_id, class_name, manual_only, summary_only=True)
        return jsonify({
            'success': True,
            'summary': summary,
            'count': len(summary)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/annotation/clear_low_confidence', methods=['POST'])
def clear_low_confidence_annotations():
    """清理置信度低于指定值的标注"""
    data = request.json
    project_id = data.get('project_id')
    confidence_threshold = data.get('confidence_threshold', 0.45)
    class_name = data.get('class_name')

    if not project_id:
        return jsonify({'success': False, 'error': '缺少项目ID'})

    try:
        manager = get_db_manager()
        deleted_count = manager.clear_low_confidence_annotations(project_id, confidence_threshold, class_name)
        
        class_info = f"类别 '{class_name}' 的" if class_name else "所有"
        return jsonify({
            'success': True,
            'deleted_count': deleted_count,
            'message': f'已清理 {class_info}低置信度标注（<{confidence_threshold}）：删除 {deleted_count} 个'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/annotation/stats_by_class', methods=['POST'])
def get_annotation_stats_by_class():
    """获取各类别的标注统计信息"""
    data = request.json
    project_id = data.get('project_id')

    if not project_id:
        return jsonify({'success': False, 'error': '缺少项目ID'})

    try:
        manager = get_db_manager()
        stats = manager.get_annotation_stats_by_class(project_id)
        return jsonify({
            'success': True,
            'stats': stats
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# ==================== 类别管理API ====================

@app.route('/api/classes/update', methods=['POST'])
def update_classes():
    """更新类别列表"""
    data = request.json
    project_id = data.get('project_id')
    classes = data.get('classes', [])

    try:
        manager = get_db_manager()
        manager.update_classes(project_id, classes)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# ==================== 导出API ====================

@app.route('/api/export/yolo', methods=['POST'])
def export_yolo():
    """导出YOLO格式"""
    data = request.json
    project_id = data.get('project_id')
    output_dir = data.get('output_dir', '')
    smooth_level = data.get('smooth_level', 'medium')
    export_type = data.get('export_type', 'segment')  # 'detect' 或 'segment'

    try:
        manager = get_db_manager()
        # 使用带标注的完整项目数据
        project = manager.get_project_with_annotations(project_id)

        if not project:
            return jsonify({'success': False, 'error': '项目不存在'})

        exporter = YOLOExporter()
        result = exporter.export(
            project, output_dir,
            format_type=export_type,
            smooth_level=smooth_level
        )
        return jsonify({'success': True, 'result': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/export/coco', methods=['POST'])
def export_coco():
    """导出COCO格式"""
    data = request.json
    project_id = data.get('project_id')
    output_dir = data.get('output_dir', '')
    smooth_level = data.get('smooth_level', 'medium')
    export_type = data.get('export_type', 'segment')  # 'detect' 或 'segment'

    try:
        manager = get_db_manager()
        # 使用带标注的完整项目数据
        project = manager.get_project_with_annotations(project_id)

        if not project:
            return jsonify({'success': False, 'error': '项目不存在'})

        exporter = COCOExporter()
        result = exporter.export(
            project, output_dir,
            export_type=export_type,
            smooth_level=smooth_level
        )
        return jsonify({'success': True, 'result': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# ==================== 导出预览API ====================

@app.route('/api/export/preview', methods=['POST'])
def export_preview():
    """生成导出预览图片，显示平滑后的分割覆盖效果"""
    import cv2
    import numpy as np
    import base64
    from io import BytesIO

    data = request.json
    project_id = data.get('project_id')
    image_index = data.get('image_index', 0)
    smooth_level = data.get('smooth_level', 'medium')
    show_polygon = data.get('show_polygon', True)
    show_fill = data.get('show_fill', True)
    opacity = data.get('opacity', 0.4)

    try:
        manager = get_db_manager()
        project = manager.get_project(project_id)
        if not project:
            return jsonify({'success': False, 'error': '项目不存在'})

        images = project.get('images', [])
        if image_index >= len(images):
            return jsonify({'success': False, 'error': '图片索引超出范围'})

        img_info = images[image_index]
        image_path = img_info.get('path')

        if not image_path or not os.path.exists(image_path):
            return jsonify({'success': False, 'error': '图片文件不存在'})

        # 读取原始图片
        img = cv2.imread(image_path)
        if img is None:
            return jsonify({'success': False, 'error': '无法读取图片'})

        overlay = img.copy()
        # 获取图片的标注数据
        annotations = manager.get_annotations(project_id, image_index)

        # 使用导出器的平滑方法
        exporter = YOLOExporter()

        # 颜色列表（BGR格式）
        colors = [
            (0, 255, 0),    # 绿色
            (255, 0, 0),    # 蓝色
            (0, 0, 255),    # 红色
            (255, 255, 0),  # 青色
            (255, 0, 255),  # 品红
            (0, 255, 255),  # 黄色
            (128, 0, 255),  # 紫色
            (255, 128, 0),  # 橙色
        ]

        for i, ann in enumerate(annotations):
            polygon = ann.get('polygon', [])
            if not polygon or len(polygon) < 3:
                continue

            # 应用平滑处理
            smoothed_polygon = exporter.smooth_polygon(polygon, smooth_level)

            # 转换为numpy数组
            pts = np.array(smoothed_polygon, dtype=np.int32)
            color = colors[i % len(colors)]

            # 绘制填充
            if show_fill:
                cv2.fillPoly(overlay, [pts], color)

            # 绘制轮廓线
            if show_polygon:
                cv2.polylines(img, [pts], True, color, 2)

        # 混合原图和覆盖层
        if show_fill:
            img = cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0)

        # 添加标注信息文字
        for i, ann in enumerate(annotations):
            polygon = ann.get('polygon', [])
            if not polygon:
                continue

            smoothed_polygon = exporter.smooth_polygon(polygon, smooth_level)
            if smoothed_polygon:
                # 计算中心点
                pts = np.array(smoothed_polygon)
                cx = int(pts[:, 0].mean())
                cy = int(pts[:, 1].mean())

                label = ann.get('class_name') or ann.get('label', '')
                color = colors[i % len(colors)]

                # 绘制标签背景
                (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(img, (cx - 2, cy - text_h - 4), (cx + text_w + 2, cy + 2), color, -1)
                cv2.putText(img, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 转换为base64
        _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        # 统计信息
        stats = {
            'total_annotations': len(annotations),
            'smooth_level': smooth_level,
            'image_size': [img.shape[1], img.shape[0]],
            'filename': img_info.get('filename', '')
        }

        return jsonify({
            'success': True,
            'preview': f'data:image/jpeg;base64,{img_base64}',
            'stats': stats
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        # 更新进度为错误
        with batch_progress_lock:
            if task_id in batch_progress:
                batch_progress[task_id]['status'] = 'error'
                batch_progress[task_id]['message'] = str(e)
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/export/preview_compare', methods=['POST'])
def export_preview_compare():
    """生成多个平滑级别的对比预览"""
    import cv2
    import numpy as np
    import base64

    data = request.json
    project_id = data.get('project_id')
    image_index = data.get('image_index', 0)
    annotation_index = data.get('annotation_index', 0)  # 指定要预览的标注索引

    try:
        manager = get_db_manager()
        project = manager.get_project(project_id)
        if not project:
            return jsonify({'success': False, 'error': '项目不存在'})

        images = project.get('images', [])
        if image_index >= len(images):
            return jsonify({'success': False, 'error': '图片索引超出范围'})

        img_info = images[image_index]
        image_path = img_info.get('path')
        annotations = img_info.get('annotations', [])

        if annotation_index >= len(annotations):
            return jsonify({'success': False, 'error': '标注索引超出范围'})

        if not image_path or not os.path.exists(image_path):
            return jsonify({'success': False, 'error': '图片文件不存在'})

        # 读取原始图片
        original_img = cv2.imread(image_path)
        if original_img is None:
            return jsonify({'success': False, 'error': '无法读取图片'})

        exporter = YOLOExporter()
        polygon = annotations[annotation_index].get('polygon', [])

        if not polygon or len(polygon) < 3:
            return jsonify({'success': False, 'error': '标注没有有效的多边形数据'})

        # 生成不同平滑级别的预览
        levels = ['none', 'low', 'medium', 'high', 'ultra']
        previews = {}

        for level in levels:
            img = original_img.copy()
            smoothed_polygon = exporter.smooth_polygon(polygon, level)
            pts = np.array(smoothed_polygon, dtype=np.int32)

            # 绘制填充和轮廓
            overlay = img.copy()
            cv2.fillPoly(overlay, [pts], (0, 255, 0))
            img = cv2.addWeighted(overlay, 0.4, img, 0.6, 0)
            cv2.polylines(img, [pts], True, (0, 255, 0), 2)

            # 添加级别标签
            cv2.putText(img, f'{level} ({len(smoothed_polygon)} pts)',
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # 转换为base64
            _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            previews[level] = f'data:image/jpeg;base64,{base64.b64encode(buffer).decode("utf-8")}'

        return jsonify({
            'success': True,
            'previews': previews,
            'original_points': len(polygon),
            'annotation_label': annotations[annotation_index].get('class_name', '')
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        # 更新进度为错误
        with batch_progress_lock:
            if task_id in batch_progress:
                batch_progress[task_id]['status'] = 'error'
                batch_progress[task_id]['message'] = str(e)
        return jsonify({'success': False, 'error': str(e), 'task_id': task_id})



# ==================== 视频分割API ====================

@app.route('/api/video/start_session', methods=['POST'])
def video_start_session():
    """开始视频分割会话"""
    data = request.json
    video_path = data.get('video_path')

    # 安全验证视频路径
    is_valid, error_msg = validate_video_path(video_path)
    if not is_valid:
        return jsonify({'success': False, 'error': error_msg})

    try:
        service = get_sam3_service()
        session_id = service.start_video_session(video_path)
        return jsonify({'success': True, 'session_id': session_id})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/video/add_prompt', methods=['POST'])
def video_add_prompt():
    """添加视频分割提示"""
    data = request.json
    session_id = data.get('session_id')
    frame_index = data.get('frame_index', 0)
    prompt_type = data.get('prompt_type', 'text')
    prompt_data = data.get('prompt_data')

    try:
        service = get_sam3_service()
        results = service.add_video_prompt(
            session_id, frame_index, prompt_type, prompt_data
        )
        return jsonify({'success': True, 'results': results})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/video/propagate', methods=['POST'])
def video_propagate():
    """传播视频分割"""
    data = request.json
    session_id = data.get('session_id')

    try:
        service = get_sam3_service()
        results = service.propagate_video(session_id)
        return jsonify({'success': True, 'results': results})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/video/close_session', methods=['POST'])
def video_close_session():
    """关闭视频会话"""
    data = request.json
    session_id = data.get('session_id')

    try:
        service = get_sam3_service()
        service.close_video_session(session_id)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# ==================== AI翻译API ====================

@app.route('/api/ai/translate', methods=['POST'])
def ai_translate():
    """
    使用OpenAI格式的API将中文翻译成简短的英文
    支持第三方API（如DeepSeek、通义千问、Moonshot等）
    """
    data = request.json
    text = data.get('text', '').strip()
    api_url = data.get('api_url', '').strip()
    api_key = data.get('api_key', '').strip()
    model = data.get('model', 'gpt-3.5-turbo').strip()

    if not text:
        return jsonify({'success': False, 'error': '文本为空'})

    if not api_url or not api_key:
        return jsonify({'success': False, 'error': 'API未配置'})

    # 确保API URL以/v1/chat/completions结尾
    if not api_url.endswith('/v1/chat/completions'):
        api_url = api_url.rstrip('/')
        if not api_url.endswith('/v1'):
            api_url += '/v1'
        api_url += '/chat/completions'

    try:
        # 构建翻译提示
        system_prompt = """You are a translation assistant for image segmentation tasks.
Translate the user's Chinese text into simple, concise English words or short phrases that can be used as object detection prompts.
Rules:
1. Output ONLY the English translation, nothing else
2. Keep it as short as possible (1-3 words preferred)
3. Use common object names (e.g., "apple", "car", "person", "red ball")
4. If multiple objects, separate with comma
5. No explanations, no quotes, just the words"""

        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}'
        }

        payload = {
            'model': model,
            'messages': [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': text}
            ],
            'max_tokens': 100,
            'temperature': 0.3
        }

        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        print(f"[AI翻译] 正在连接: {api_url}")

        response = requests.post(
            api_url,
            headers=headers,
            json=payload,
            timeout=30,
            verify=False  # 跳过SSL证书验证，解决WSL环境下的证书问题
        )

        print(f"[AI翻译] 响应状态码: {response.status_code}")

        if response.status_code != 200:
            error_msg = f'API请求失败: {response.status_code}'
            try:
                error_data = response.json()
                if 'error' in error_data:
                    error_msg = error_data['error'].get('message', error_msg)
            except:
                pass
            return jsonify({'success': False, 'error': error_msg})

        result = response.json()
        translated = result['choices'][0]['message']['content'].strip()

        print(f"[AI翻译] {text} -> {translated}")

        return jsonify({
            'success': True,
            'original': text,
            'translated': translated
        })

    except requests.exceptions.Timeout:
        return jsonify({'success': False, 'error': 'API请求超时 (30秒)'})
    except requests.exceptions.SSLError as e:
        print(f"[AI翻译] SSL错误: {e}")
        return jsonify({'success': False, 'error': f'SSL证书错误'})
    except requests.exceptions.ConnectionError as e:
        print(f"[AI翻译] 连接错误: {e}")
        return jsonify({'success': False, 'error': '无法连接到API服务器'})
    except Exception as e:
        print(f"[AI翻译错误] {e}")
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/ai/test', methods=['POST'])
def ai_test():
    """测试AI API配置是否有效"""
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    data = request.json
    api_url = data.get('api_url', '').strip()
    api_key = data.get('api_key', '').strip()
    model = data.get('model', 'gpt-3.5-turbo').strip()

    if not api_url or not api_key:
        return jsonify({'success': False, 'error': 'API地址和密钥不能为空'})

    # 确保API URL格式正确
    if not api_url.endswith('/v1/chat/completions'):
        api_url = api_url.rstrip('/')
        if not api_url.endswith('/v1'):
            api_url += '/v1'
        api_url += '/chat/completions'

    try:
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}'
        }

        payload = {
            'model': model,
            'messages': [
                {'role': 'user', 'content': 'Hello'}
            ],
            'max_tokens': 10
        }

        print(f"[AI测试] 正在连接: {api_url}")

        response = requests.post(
            api_url,
            headers=headers,
            json=payload,
            timeout=30,
            verify=False  # 跳过SSL证书验证，解决WSL环境下的证书问题
        )

        print(f"[AI测试] 响应状态码: {response.status_code}")

        if response.status_code == 200:
            return jsonify({'success': True, 'message': 'API连接成功'})
        else:
            error_msg = f'状态码: {response.status_code}'
            try:
                error_data = response.json()
                if 'error' in error_data:
                    error_msg = error_data['error'].get('message', error_msg)
            except:
                pass
            return jsonify({'success': False, 'error': error_msg})

    except requests.exceptions.Timeout:
        return jsonify({'success': False, 'error': '连接超时 (30秒)'})
    except requests.exceptions.SSLError as e:
        print(f"[AI测试] SSL错误: {e}")
        return jsonify({'success': False, 'error': f'SSL证书错误: {str(e)[:100]}'})
    except requests.exceptions.ConnectionError as e:
        print(f"[AI测试] 连接错误: {e}")
        return jsonify({'success': False, 'error': f'无法连接到API服务器，请检查网络或API地址是否正确'})
    except Exception as e:
        print(f"[AI测试] 未知错误: {e}")
        return jsonify({'success': False, 'error': str(e)})


def wait_for_server(url, timeout=30):
    """等待服务器启动就绪"""
    import time
    import urllib.request
    import urllib.error

    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            urllib.request.urlopen(url, timeout=1)
            return True
        except (urllib.error.URLError, urllib.error.HTTPError):
            time.sleep(0.3)
    return False


def open_browser(url):
    """等待服务就绪后打开浏览器（独立窗口模式）"""
    print("[INFO] 等待服务启动...")

    # 等待服务就绪
    if not wait_for_server(url):
        print("[ERROR] 服务启动超时，请手动打开浏览器访问:", url)
        return

    print("[INFO] 服务已就绪，正在打开浏览器...")

    # 尝试不同的浏览器路径
    chrome_paths = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    ]

    browser_path = None
    for path in chrome_paths:
        if os.path.exists(path):
            browser_path = path
            break

    if browser_path:
        # 使用 --app 模式打开，类似独立应用（无地址栏）
        subprocess.Popen([
            browser_path,
            f'--app={url}',
            '--disable-infobars',
            '--no-first-run',
            '--force-device-scale-factor=1',  # 强制缩放比例为1，避免字体变小
        ])
        print(f"[INFO] 已在独立窗口中打开: {url}")
    else:
        webbrowser.open(url)
        print(f"[INFO] 已在默认浏览器中打开: {url}")


# NMS配置API
@app.route('/api/config/nms', methods=['POST'])
def update_nms_config():
    """更新NMS配置（运行时）"""
    try:
        data = request.json
        if not data:
            return jsonify({'success': False, 'error': '请求数据为空'})

        # 更新配置
        SegmentationConfig.ENABLE_NMS = data.get('enabled', True)
        SegmentationConfig.NMS_IOU_THRESHOLD = data.get('iou_threshold', 0.4)
        SegmentationConfig.NMS_OVERLAP_MODE = data.get('overlap_mode', 'iou')
        SegmentationConfig.NMS_MIN_AREA_RATIO = data.get('min_area_ratio', 0.1)
        SegmentationConfig.NMS_MASK_LEVEL = data.get('mask_level', True)

        # 重新初始化服务以应用新配置
        try:
            reset_sam3_service()
            reset_sam3_concurrent()
        except Exception as e:
            print(f"[WARNING] 重新初始化服务时出错: {e}")
            # 即使重置失败也继续,配置已更新

        print(f"[NMS配置更新] enabled={SegmentationConfig.ENABLE_NMS}, "
              f"iou={SegmentationConfig.NMS_IOU_THRESHOLD}, "
              f"mode={SegmentationConfig.NMS_OVERLAP_MODE}")

        return jsonify({'success': True, 'config': {
            'enabled': SegmentationConfig.ENABLE_NMS,
            'iou_threshold': SegmentationConfig.NMS_IOU_THRESHOLD,
            'overlap_mode': SegmentationConfig.NMS_OVERLAP_MODE,
            'min_area_ratio': SegmentationConfig.NMS_MIN_AREA_RATIO,
            'mask_level': SegmentationConfig.NMS_MASK_LEVEL
        }})
    except Exception as e:
        print(f"[ERROR] 更新NMS配置失败: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/config/nms', methods=['GET'])
def get_nms_config():
    """获取当前NMS配置"""
    try:
        return jsonify({'success': True, 'config': {
            'enabled': SegmentationConfig.ENABLE_NMS,
            'iou_threshold': SegmentationConfig.NMS_IOU_THRESHOLD,
            'overlap_mode': SegmentationConfig.NMS_OVERLAP_MODE,
            'min_area_ratio': SegmentationConfig.NMS_MIN_AREA_RATIO,
            'mask_level': SegmentationConfig.NMS_MASK_LEVEL
        }})
    except Exception as e:
        print(f"[ERROR] 获取NMS配置失败: {e}")
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/app/exit', methods=['POST'])
def exit_app():
    """退出程序"""
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        # 强制退出
        os._exit(0)
    func()
    return jsonify({'success': True})


if __name__ == '__main__':
    print("=" * 50)
    print("SAM3 AN - 数据标注工具")
    print("=" * 50)

    # 在后台线程中等待服务就绪后打开浏览器
    url = "http://localhost:5001"
    threading.Thread(target=open_browser, args=(url,), daemon=True).start()

    print(f"[INFO] 正在启动服务器...")
    print(f"[INFO] 服务就绪后将自动打开浏览器")
    print("=" * 50)

    # 启动Flask服务器（关闭debug模式以避免重复打开浏览器）
    # 设置请求超时时间为 10 分钟，以支持大批量处理
    from werkzeug.serving import WSGIRequestHandler
    WSGIRequestHandler.protocol_version = "HTTP/1.1"
    app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)
