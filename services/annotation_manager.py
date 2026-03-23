"""
标注数据管理器
负责项目、图片、标注的增删改查
"""
import os
import json
from pathlib import Path
from datetime import datetime
import uuid
import threading


class AnnotationManager:
    """标注数据管理器"""

    def __init__(self, data_dir: str = None):
        if data_dir is None:
            data_dir = Path(__file__).parent.parent / 'data'
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

        self.projects = {}
        self._lock = threading.Lock()
        self._load_all_projects()

    def _load_all_projects(self):
        """加载所有项目"""
        projects_file = self.data_dir / 'projects.json'
        if projects_file.exists():
            with open(projects_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.projects = {p['id']: p for p in data.get('projects', [])}

    def _save_all_projects(self):
        """保存所有项目"""
        projects_file = self.data_dir / 'projects.json'
        with open(projects_file, 'w', encoding='utf-8') as f:
            json.dump({
                'projects': list(self.projects.values()),
                'updated_at': datetime.now().isoformat()
            }, f, ensure_ascii=False, indent=2)

    def create_project(self, project: dict) -> dict:
        """创建新项目"""
        with self._lock:
            project_id = project.get('id', str(uuid.uuid4())[:8])
            project['id'] = project_id
            project['created_at'] = datetime.now().isoformat()
            project['updated_at'] = datetime.now().isoformat()

            if 'images' not in project:
                project['images'] = []
            if 'classes' not in project:
                project['classes'] = []

            self.projects[project_id] = project
            self._save_all_projects()

            # 创建项目专属目录
            project_dir = self.data_dir / project_id
            project_dir.mkdir(exist_ok=True)

            return project

    def get_project(self, project_id: str) -> dict:
        """获取项目"""
        return self.projects.get(project_id)

    def list_projects(self) -> list:
        """列出所有项目"""
        return list(self.projects.values())

    def update_project(self, project_id: str, updates: dict) -> dict:
        """更新项目"""
        with self._lock:
            if project_id not in self.projects:
                raise ValueError(f"项目不存在: {project_id}")

            project = self.projects[project_id]
            project.update(updates)
            project['updated_at'] = datetime.now().isoformat()

            self._save_all_projects()
            return project

    def delete_project(self, project_id: str):
        """删除项目"""
        with self._lock:
            if project_id in self.projects:
                del self.projects[project_id]
                self._save_all_projects()

                # 删除项目目录
                project_dir = self.data_dir / project_id
                if project_dir.exists():
                    import shutil
                    shutil.rmtree(project_dir)

    def update_project_images(self, project_id: str, images: list, image_dir: str):
        """更新项目图片列表"""
        with self._lock:
            if project_id not in self.projects:
                raise ValueError(f"项目不存在: {project_id}")

            project = self.projects[project_id]

            # 保留已有标注
            existing_annotations = {}
            for img in project.get('images', []):
                if img.get('annotations'):
                    existing_annotations[img['filename']] = img['annotations']

            # 更新图片列表，保留已有标注
            for img in images:
                filename = img['filename']
                if filename in existing_annotations:
                    img['annotations'] = existing_annotations[filename]
                    img['annotated'] = True

            project['images'] = images
            project['image_dir'] = image_dir
            project['updated_at'] = datetime.now().isoformat()

            self._save_all_projects()
            self._save_project_annotations(project_id)

    def add_annotations(self, project_id: str, image_index: int,
                        annotations: list, label: str = None):
        """添加标注（来自SAM3分割结果）
        先删除指定类别的非手动标注，再添加新标注
        """
        with self._lock:
            if project_id not in self.projects:
                raise ValueError(f"项目不存在: {project_id}")

            project = self.projects[project_id]
            if image_index >= len(project['images']):
                raise ValueError(f"图片索引越界: {image_index}")

            image = project['images'][image_index]

            # 确定目标类别名称
            target_class = label
            if annotations and annotations[0].get('class_name'):
                target_class = annotations[0].get('class_name')

            # 先删除指定类别的非手动标注（保留手动标注和其他类别）
            if 'annotations' in image and target_class:
                filtered_annotations = []
                for ann in image['annotations']:
                    ann_class = ann.get('class_name') or ann.get('label', '')
                    ann_manual = ann.get('manual', False)

                    # 保留手动标注，或不同类别的标注
                    if ann_manual or ann_class != target_class:
                        filtered_annotations.append(ann)
                    else:
                        # 删除该类别的非手动标注
                        print(f"[JSON] 删除类别 '{target_class}' 的非手动标注")

                image['annotations'] = filtered_annotations

            # 初始化标注数组
            if 'annotations' not in image:
                image['annotations'] = []

            # 为每个标注添加类别标签
            for ann in annotations:
                if label:
                    ann['class_name'] = label
                if 'id' not in ann:
                    ann['id'] = str(uuid.uuid4())[:8]
                # 批量分割的标注默认为非手动
                if 'manual' not in ann:
                    ann['manual'] = False

            # 添加新标注
            image['annotations'].extend(annotations)
            # 更新标注状态：只有当有标注时才标记为已标注
            image['annotated'] = len(image['annotations']) > 0

            project['updated_at'] = datetime.now().isoformat()
            self._save_all_projects()
            self._save_project_annotations(project_id)

    def add_annotations_batch(self, annotations_data: list):
        """批量添加标注（高性能版本，先删除指定类别的非手动标注）

        Args:
            annotations_data: [(project_id, image_index, annotations, label), ...]
        """
        with self._lock:
            # 处理每张图片
            for project_id, image_index, annotations, label in annotations_data:
                if project_id not in self.projects:
                    print(f"[JSON] 项目不存在: {project_id}, 跳过")
                    continue

                project = self.projects[project_id]
                if image_index >= len(project['images']):
                    print(f"[JSON] 图片索引越界: {image_index}, 跳过")
                    continue

                image = project['images'][image_index]

                # 确定目标类别名称
                target_class = label
                if annotations and annotations[0].get('class_name'):
                    target_class = annotations[0].get('class_name')

                # 先删除指定类别的非手动标注（保留手动标注和其他类别）
                if 'annotations' in image and target_class:
                    filtered_annotations = []
                    for ann in image['annotations']:
                        ann_class = ann.get('class_name') or ann.get('label', '')
                        ann_manual = ann.get('manual', False)

                        # 保留手动标注，或不同类别的标注
                        if ann_manual or ann_class != target_class:
                            filtered_annotations.append(ann)
                        else:
                            # 删除该类别的非手动标注
                            pass

                    image['annotations'] = filtered_annotations

                # 初始化标注数组
                if 'annotations' not in image:
                    image['annotations'] = []

                # 为每个标注添加类别标签
                for ann in annotations:
                    if label:
                        ann['class_name'] = label
                    if 'id' not in ann:
                        ann['id'] = str(uuid.uuid4())[:8]
                    # 批量分割的标注默认为非手动
                    if 'manual' not in ann:
                        ann['manual'] = False

                # 添加新标注
                image['annotations'].extend(annotations)
                # 更新标注状态：只有当有标注时才标记为已标注
                image['annotated'] = len(image['annotations']) > 0

                project['updated_at'] = datetime.now().isoformat()

            # 保存所有项目
            self._save_all_projects()
            # 保存每个项目的标注
            for project_id, _, _, _ in annotations_data:
                if project_id in self.projects:
                    self._save_project_annotations(project_id)

    def save_annotations(self, project_id: str, image_index: int, annotations: list):
        """保存标注（覆盖）"""
        with self._lock:
            if project_id not in self.projects:
                raise ValueError(f"项目不存在: {project_id}")

            project = self.projects[project_id]
            if image_index >= len(project['images']):
                raise ValueError(f"图片索引越界: {image_index}")

            image = project['images'][image_index]
            image['annotations'] = annotations
            image['annotated'] = len(annotations) > 0

            project['updated_at'] = datetime.now().isoformat()
            self._save_all_projects()
            self._save_project_annotations(project_id)

    def get_annotations(self, project_id: str, image_index: int) -> list:
        """获取标注"""
        project = self.projects.get(project_id)
        if not project:
            return []

        images = project.get('images', [])
        if image_index >= len(images):
            return []

        return images[image_index].get('annotations', [])

    def update_annotation(self, project_id: str, image_index: int,
                          annotation_id: str, updates: dict):
        """更新单个标注"""
        with self._lock:
            if project_id not in self.projects:
                raise ValueError(f"项目不存在: {project_id}")

            project = self.projects[project_id]
            if image_index >= len(project['images']):
                raise ValueError(f"图片索引越界: {image_index}")

            image = project['images'][image_index]
            annotations = image.get('annotations', [])

            for ann in annotations:
                if ann.get('id') == annotation_id:
                    ann.update(updates)
                    break

            project['updated_at'] = datetime.now().isoformat()
            self._save_all_projects()
            self._save_project_annotations(project_id)

    def delete_annotation(self, project_id: str, image_index: int, annotation_id: str):
        """删除单个标注"""
        with self._lock:
            if project_id not in self.projects:
                raise ValueError(f"项目不存在: {project_id}")

            project = self.projects[project_id]
            if image_index >= len(project['images']):
                raise ValueError(f"图片索引越界: {image_index}")

            image = project['images'][image_index]
            annotations = image.get('annotations', [])

            image['annotations'] = [
                ann for ann in annotations if ann.get('id') != annotation_id
            ]
            image['annotated'] = len(image['annotations']) > 0

            project['updated_at'] = datetime.now().isoformat()
            self._save_all_projects()
            self._save_project_annotations(project_id)

    def update_classes(self, project_id: str, classes: list):
        """更新类别列表"""
        with self._lock:
            if project_id not in self.projects:
                raise ValueError(f"项目不存在: {project_id}")

            project = self.projects[project_id]
            project['classes'] = classes
            project['updated_at'] = datetime.now().isoformat()

            self._save_all_projects()

    def _save_project_annotations(self, project_id: str):
        """保存项目标注到单独文件"""
        project = self.projects.get(project_id)
        if not project:
            return

        project_dir = self.data_dir / project_id
        project_dir.mkdir(exist_ok=True)

        annotations_file = project_dir / 'annotations.json'
        with open(annotations_file, 'w', encoding='utf-8') as f:
            json.dump({
                'project_id': project_id,
                'images': project.get('images', []),
                'classes': project.get('classes', []),
                'updated_at': datetime.now().isoformat()
            }, f, ensure_ascii=False, indent=2)

    def mark_image_annotated(self, project_id: str, image_index: int, annotated: bool = True):
        """标记图片为已标注/未标注"""
        with self._lock:
            if project_id not in self.projects:
                raise ValueError(f"项目不存在: {project_id}")

            project = self.projects[project_id]
            if image_index >= len(project['images']):
                raise ValueError(f"图片索引越界: {image_index}")

            project['images'][image_index]['annotated'] = annotated
            project['updated_at'] = datetime.now().isoformat()

            self._save_all_projects()

    def get_annotation_stats(self, project_id: str) -> dict:
        """获取标注统计"""
        project = self.projects.get(project_id)
        if not project:
            return {}

        images = project.get('images', [])
        total = len(images)
        annotated = sum(1 for img in images if img.get('annotated', False))
        total_annotations = sum(
            len(img.get('annotations', [])) for img in images
        )

        return {
            'total_images': total,
            'annotated_images': annotated,
            'unannotated_images': total - annotated,
            'total_annotations': total_annotations,
            'progress': annotated / total * 100 if total > 0 else 0
        }
