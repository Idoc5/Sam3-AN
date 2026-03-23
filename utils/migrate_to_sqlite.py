"""
数据迁移脚本：JSON → SQLite
将现有的JSON格式标注数据迁移到SQLite数据库
"""
import sys
import json
from pathlib import Path
from datetime import datetime

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.db_annotation_manager import DBAnnotationManager
from services.annotation_manager import AnnotationManager


def backup_json_data():
    """备份JSON数据"""
    data_dir = Path(__file__).parent.parent / 'data'
    backup_dir = data_dir / 'backup_json'

    if backup_dir.exists():
        backup_dir = data_dir / f'backup_json_{datetime.now().strftime("%Y%m%d_%H%M%S")}'

    backup_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("备份JSON数据")
    print("=" * 60)
    print(f"备份目录: {backup_dir}")

    # 备份projects.json
    projects_file = data_dir / 'projects.json'
    if projects_file.exists():
        import shutil
        shutil.copy2(projects_file, backup_dir / 'projects.json')
        print(f"✓ 备份 projects.json")

    # 备份项目目录
    for project_dir in data_dir.iterdir():
        if project_dir.is_dir() and len(project_dir.name) == 8:  # 假设项目ID是8位
            target_dir = backup_dir / project_dir.name
            if not target_dir.exists():
                import shutil
                shutil.copytree(project_dir, target_dir)
                print(f"✓ 备份项目目录: {project_dir.name}")

    print(f"备份完成: {backup_dir}")
    return backup_dir


def migrate_json_to_sqlite():
    """将JSON数据迁移到SQLite"""
    print("\n" + "=" * 60)
    print("开始数据迁移: JSON → SQLite")
    print("=" * 60)

    # 备份现有数据
    backup_dir = backup_json_data()

    # 初始化管理器
    json_manager = AnnotationManager()
    db_manager = DBAnnotationManager()

    # 迁移项目
    projects = json_manager.list_projects()
    print(f"\n发现 {len(projects)} 个项目\n")

    total_images = 0
    total_annotations = 0
    migrated_projects = []

    for idx, project in enumerate(projects, 1):
        print(f"[{idx}/{len(projects)}] 迁移项目: {project['name']} (ID: {project['id']})")

        try:
            # 创建项目
            db_project = db_manager.create_project(project)
            print(f"  ✓ 创建项目")

            # 批量添加图片
            images = []
            project_images = project.get('images', [])

            for img_idx, img in enumerate(project_images, 1):
                images.append({
                    'filename': img['filename'],
                    'path': img.get('path'),
                    'annotated': img.get('annotated', False),
                    'index_in_project': img_idx - 1
                })

            if images:
                db_manager.add_images_batch(project['id'], images)
                print(f"  ✓ 添加 {len(images)} 张图片")
                total_images += len(images)

            # 迁移标注
            annotations_data = []
            project_annotations = 0

            for img_idx, img in enumerate(project_images):
                annotations = img.get('annotations', [])
                if annotations:
                    annotations_data.append((
                        project['id'],
                        img_idx,
                        annotations,
                        None
                    ))
                    project_annotations += len(annotations)

            if annotations_data:
                db_manager.add_annotations_batch(annotations_data)
                print(f"  ✓ 添加 {project_annotations} 个标注")
                total_annotations += project_annotations

            # 验证迁移结果
            stats = db_manager.get_annotation_stats(project['id'])
            print(f"  ✓ 验证: {stats['total_images']} 张图片, {stats['total_annotations']} 个标注")

            migrated_projects.append(project['id'])

        except Exception as e:
            print(f"  ✗ 迁移失败: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 打印统计信息
    print("\n" + "=" * 60)
    print("数据迁移完成!")
    print("=" * 60)
    print(f"迁移项目数: {len(migrated_projects)}/{len(projects)}")
    print(f"总图片数: {total_images}")
    print(f"总标注数: {total_annotations}")

    # 数据库统计
    db_stats = db_manager.get_db_stats()
    print(f"数据库文件大小: {db_stats.get('db_size_mb', 0)} MB")

    print(f"\n数据库路径: {db_manager.db_path}")
    print(f"数据备份目录: {backup_dir}")

    print("\n" + "=" * 60)
    print("迁移验证")
    print("=" * 60)

    # 验证数据一致性
    all_valid = True
    for project_id in migrated_projects[:3]:  # 验证前3个项目
        json_project = json_manager.get_project(project_id)
        db_project = db_manager.get_project(project_id)

        json_images = len(json_project.get('images', []))
        db_images = len(db_project.get('images', []))

        json_annotations = sum(
            len(img.get('annotations', []))
            for img in json_project.get('images', [])
        )
        db_annotations = db_manager.get_annotation_stats(project_id)['total_annotations']

        print(f"\n项目 {project_id}:")
        print(f"  图片: JSON={json_images}, DB={db_images}, {'✓' if json_images == db_images else '✗'}")
        print(f"  标注: JSON={json_annotations}, DB={db_annotations}, {'✓' if json_annotations == db_annotations else '✗'}")

        if json_images != db_images or json_annotations != db_annotations:
            all_valid = False

    if all_valid:
        print("\n✓ 数据验证通过!")
    else:
        print("\n⚠️ 发现数据不一致，请检查日志")

    print("=" * 60)

    return migrated_projects, backup_dir


if __name__ == '__main__':
    migrated_projects, backup_dir = migrate_json_to_sqlite()

    print(f"\n迁移完成!")
    print(f"备份目录: {backup_dir}")
    print(f"如需回滚，请从备份目录恢复JSON文件")
