"""
SQLite数据库标注数据管理器
替代JSON存储，提供高性能的批量操作和并发支持
"""
import sqlite3
import json
import threading
from pathlib import Path
from contextlib import contextmanager
from typing import List, Dict, Optional
from datetime import datetime
import uuid


class DBAnnotationManager:
    """基于SQLite的标注数据管理器"""

    def __init__(self, db_path: str = None, timeout: float = 30.0):
        if db_path is None:
            db_path = Path(__file__).parent.parent / 'data' / 'annotations.db'
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # 连接池（线程安全）
        self._local = threading.local()
        self.timeout = timeout

        # 初始化数据库
        self._init_db()

    @contextmanager
    def _get_conn(self):
        """获取数据库连接（线程安全）"""
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self._local.conn = sqlite3.connect(
                self.db_path,
                timeout=self.timeout,
                check_same_thread=False
            )
            self._local.conn.row_factory = sqlite3.Row

            # 优化SQLite性能
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn.execute("PRAGMA cache_size=-64000")
            self._local.conn.execute("PRAGMA temp_store=MEMORY")

        conn = self._local.conn
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def _init_db(self):
        """初始化数据库表结构"""
        with self._get_conn() as conn:
            # 创建项目表
            conn.execute("""
                CREATE TABLE IF NOT EXISTS projects (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    image_dir TEXT,
                    output_dir TEXT,
                    export_format TEXT DEFAULT 'yolo',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # 创建类别表
            conn.execute("""
                CREATE TABLE IF NOT EXISTS classes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE,
                    UNIQUE(project_id, name)
                )
            """)

            # 创建图片表
            conn.execute("""
                CREATE TABLE IF NOT EXISTS images (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    path TEXT,
                    annotated INTEGER DEFAULT 0,
                    index_in_project INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE,
                    UNIQUE(project_id, index_in_project)
                )
            """)

            # 创建标注表
            conn.execute("""
                CREATE TABLE IF NOT EXISTS annotations (
                    id TEXT PRIMARY KEY,
                    image_id INTEGER NOT NULL,
                    label TEXT,
                    class_name TEXT,
                    score REAL,
                    bbox TEXT,
                    polygon TEXT,
                    area REAL,
                    manual INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (image_id) REFERENCES images(id) ON DELETE CASCADE
                )
            """)

            # 创建索引
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_images_project
                ON images(project_id, index_in_project)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_annotations_image
                ON annotations(image_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_annotations_class
                ON annotations(class_name)
            """)

            # 创建触发器：自动更新项目时间戳
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS update_project_timestamp
                AFTER UPDATE ON images
                BEGIN
                    UPDATE projects SET updated_at = CURRENT_TIMESTAMP
                    WHERE id = NEW.project_id;
                END
            """)

            # 迁移：如果 annotations 表没有 manual 字段，则添加
            cursor = conn.execute("PRAGMA table_info(annotations)")
            columns = [col[1] for col in cursor.fetchall()]
            if 'manual' not in columns:
                conn.execute("ALTER TABLE annotations ADD COLUMN manual INTEGER DEFAULT 0")
                print("[DB] 已添加 manual 字段到 annotations 表")

    # ==================== 项目管理 ====================

    def create_project(self, project: dict) -> dict:
        """创建新项目"""
        project_id = project.get('id', str(uuid.uuid4())[:8])
        project['id'] = project_id
        project['created_at'] = datetime.now().isoformat()
        project['updated_at'] = datetime.now().isoformat()

        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO projects (id, name, image_dir, output_dir, export_format)
                VALUES (?, ?, ?, ?, ?)
            """, (
                project_id,
                project['name'],
                project.get('image_dir'),
                project.get('output_dir'),
                project.get('export_format', 'yolo')
            ))

            # 添加类别
            for class_name in project.get('classes', []):
                self.add_class(project_id, class_name)

            return self.get_project(project_id)

    def get_project(self, project_id: str) -> Optional[dict]:
        """获取项目（包含图片列表）"""
        with self._get_conn() as conn:
            cursor = conn.cursor()

            # 获取项目基本信息
            cursor.execute("SELECT * FROM projects WHERE id = ?", (project_id,))
            row = cursor.fetchone()
            if not row:
                return None

            project = dict(row)

            # 获取类别
            cursor.execute("""
                SELECT name FROM classes WHERE project_id = ? ORDER BY name
            """, (project_id,))
            project['classes'] = [r['name'] for r in cursor.fetchall()]

            # 获取图片列表（不包含标注详情，提升性能）
            cursor.execute("""
                SELECT id, filename, path, annotated, index_in_project
                FROM images
                WHERE project_id = ?
                ORDER BY index_in_project
            """, (project_id,))

            images = []
            for row in cursor.fetchall():
                images.append({
                    'id': row['id'],
                    'filename': row['filename'],
                    'path': row['path'],
                    'annotated': bool(row['annotated']),
                    'index': row['index_in_project']
                })

            project['images'] = images
            return project

    def get_project_with_annotations(self, project_id: str) -> Optional[dict]:
        """获取项目（包含图片列表和完整标注数据）"""
        with self._get_conn() as conn:
            cursor = conn.cursor()

            # 获取项目基本信息
            cursor.execute("SELECT * FROM projects WHERE id = ?", (project_id,))
            row = cursor.fetchone()
            if not row:
                return None

            project = dict(row)

            # 获取类别
            cursor.execute("""
                SELECT name FROM classes WHERE project_id = ? ORDER BY name
            """, (project_id,))
            project['classes'] = [r['name'] for r in cursor.fetchall()]

            # 获取图片列表（包含标注详情）
            cursor.execute("""
                SELECT id, filename, path, annotated, index_in_project
                FROM images
                WHERE project_id = ?
                ORDER BY index_in_project
            """, (project_id,))

            images = []
            for img_row in cursor.fetchall():
                image = {
                    'id': img_row['id'],
                    'filename': img_row['filename'],
                    'path': img_row['path'],
                    'annotated': bool(img_row['annotated']),
                    'index': img_row['index_in_project'],
                    'annotations': []  # 稍后填充
                }

                # 获取该图片的所有标注
                cursor.execute("""
                    SELECT id, label, class_name, score, bbox, polygon, area
                    FROM annotations WHERE image_id = ?
                """, (img_row['id'],))

                for ann_row in cursor.fetchall():
                    annotation = {
                        'id': ann_row['id'],
                        'label': ann_row['label'],
                        'class_name': ann_row['class_name'],
                        'score': ann_row['score'],
                        'bbox': json.loads(ann_row['bbox']) if ann_row['bbox'] else [],
                        'polygon': json.loads(ann_row['polygon']) if ann_row['polygon'] else [],
                        'area': ann_row['area']
                    }
                    image['annotations'].append(annotation)

                images.append(image)

            project['images'] = images
            return project

    def list_projects(self) -> list:
        """列出所有项目（轻量级，含图片统计和类别）"""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, name, image_dir, output_dir, export_format, created_at, updated_at
                FROM projects ORDER BY updated_at DESC
            """)
            projects = [dict(row) for row in cursor.fetchall()]

            # 为每个项目添加统计和类别
            for project in projects:
                # 获取图片统计
                cursor.execute("""
                    SELECT
                        COUNT(*) as total_count,
                        SUM(CASE WHEN annotated = 1 THEN 1 ELSE 0 END) as annotated_count
                    FROM images WHERE project_id = ?
                """, (project['id'],))
                stats = cursor.fetchone()
                project['image_count'] = stats['total_count'] or 0
                project['annotated_count'] = stats['annotated_count'] or 0

                # 获取类别列表
                cursor.execute("""
                    SELECT name FROM classes WHERE project_id = ? ORDER BY name
                """, (project['id'],))
                project['classes'] = [r['name'] for r in cursor.fetchall()]

            return projects

    def update_project(self, project_id: str, updates: dict) -> dict:
        """更新项目"""
        with self._get_conn() as conn:
            cursor = conn.cursor()

            # 构建更新语句
            update_fields = []
            update_values = []

            for key, value in updates.items():
                if key in ['name', 'image_dir', 'output_dir', 'export_format']:
                    update_fields.append(f"{key} = ?")
                    update_values.append(value)

            if update_fields:
                update_values.append(project_id)
                cursor.execute(f"""
                    UPDATE projects SET {', '.join(update_fields)}, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, update_values)

            return self.get_project(project_id)

    def delete_project(self, project_id: str):
        """删除项目（级联删除图片和标注）"""
        with self._get_conn() as conn:
            conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))

    # ==================== 类别管理 ====================

    def add_class(self, project_id: str, class_name: str):
        """添加类别"""
        with self._get_conn() as conn:
            conn.execute("""
                INSERT OR IGNORE INTO classes (project_id, name)
                VALUES (?, ?)
            """, (project_id, class_name))

    def update_classes(self, project_id: str, classes: list):
        """更新类别列表"""
        with self._get_conn() as conn:
            cursor = conn.cursor()

            # 删除现有类别
            cursor.execute("DELETE FROM classes WHERE project_id = ?", (project_id,))

            # 批量添加新类别
            if classes:
                cursor.executemany("""
                    INSERT INTO classes (project_id, name)
                    VALUES (?, ?)
                """, [(project_id, name) for name in classes])

    # ==================== 图片管理 ====================

    def add_images_batch(self, project_id: str, images: List[dict]):
        """批量添加图片"""
        with self._get_conn() as conn:
            cursor = conn.cursor()

            # 使用executemany批量插入
            values = []
            for img in images:
                values.append((
                    project_id,
                    img['filename'],
                    img.get('path'),
                    1 if img.get('annotated', False) else 0,
                    img['index_in_project']
                ))

            cursor.executemany("""
                INSERT OR REPLACE INTO images
                (project_id, filename, path, annotated, index_in_project)
                VALUES (?, ?, ?, ?, ?)
            """, values)

    def update_project_images(self, project_id: str, images: list, image_dir: str):
        """更新项目图片列表（保留已有标注）"""
        import os
        
        # 收集要删除的图片路径（在事务外删除）
        paths_to_delete = []
        
        with self._get_conn() as conn:
            cursor = conn.cursor()

            # 获取现有图片ID和路径（用于保留标注关联）
            cursor.execute("""
                SELECT id, filename, path, index_in_project FROM images
                WHERE project_id = ?
            """, (project_id,))

            existing_images = {row['filename']: {'id': row['id'], 'path': row['path'], 'index': row['index_in_project']}
                               for row in cursor.fetchall()}

            # 删除不在新列表中的旧图片（避免UNIQUE约束冲突）
            new_filenames = {img['filename'] for img in images}
            for filename, img_data in list(existing_images.items()):
                if filename not in new_filenames:
                    # 删除该图片的所有标注
                    cursor.execute("DELETE FROM annotations WHERE image_id = ?", (img_data['id'],))
                    # 删除图片记录
                    cursor.execute("DELETE FROM images WHERE id = ?", (img_data['id'],))
                    # 记录要删除的图片路径
                    if img_data.get('path'):
                        paths_to_delete.append(img_data['path'])
                    # 从 existing_images 中移除，避免后续处理
                    del existing_images[filename]

            # 获取现有标注数据（按filename组织）
            cursor.execute("""
                SELECT i.filename, a.id as ann_id, a.label, a.class_name, a.score, a.bbox, a.polygon, a.area, a.manual
                FROM images i
                LEFT JOIN annotations a ON i.id = a.image_id
                WHERE i.project_id = ?
            """, (project_id,))

            existing_annotations = {}
            for row in cursor.fetchall():
                filename = row['filename']
                if filename not in existing_annotations:
                    existing_annotations[filename] = {'old_image_id': existing_images.get(filename, {}).get('id'), 'annotations': []}
                if row['ann_id']:  # 有标注
                    existing_annotations[filename]['annotations'].append({
                        'id': row['ann_id'],
                        'label': row['label'],
                        'class_name': row['class_name'],
                        'score': row['score'],
                        'bbox': json.loads(row['bbox']) if row['bbox'] else [],
                        'polygon': json.loads(row['polygon']) if row['polygon'] else [],
                        'area': row['area'],
                        'manual': row['manual']
                    })

            # 批量插入或更新图片
            values = []
            for idx, img in enumerate(images):
                filename = img['filename']
                has_annotations = filename in existing_annotations and existing_annotations[filename]['annotations']

                # 如果图片已存在，保留其ID（INSERT OR REPLACE会创建新ID）
                # 所以我们需要先检查是否存在，如果存在则更新，否则插入
                if filename in existing_images:
                    cursor.execute("""
                        UPDATE images
                        SET path = ?, annotated = ?, index_in_project = ?
                        WHERE project_id = ? AND filename = ?
                    """, (img.get('path'), 1 if has_annotations else 0, idx, project_id, filename))
                    # 保留原有ID
                    existing_images[filename]['new_id'] = existing_images[filename]['id']
                else:
                    # 插入新图片
                    cursor.execute("""
                        INSERT INTO images (project_id, filename, path, annotated, index_in_project)
                        VALUES (?, ?, ?, ?, ?)
                    """, (project_id, filename, img.get('path'), 1 if has_annotations else 0, idx))
                    existing_images[filename]['new_id'] = cursor.lastrowid

            # 重新关联标注：删除旧标注，插入新标注（使用新的image_id）
            # 注意：如果图片ID没变，就不需要重新插入标注
            for filename, data in existing_annotations.items():
                old_image_id = data['old_image_id']
                new_image_id = existing_images.get(filename, {}).get('new_id')

                if not old_image_id or not new_image_id:
                    continue

                # 如果image_id没变，标注仍然有效，无需处理
                if old_image_id == new_image_id:
                    continue

                # 删除新image_id下已有的标注（避免重复）
                cursor.execute("DELETE FROM annotations WHERE image_id = ?", (new_image_id,))

                # 将标注从旧image_id转移到新image_id
                for ann in data['annotations']:
                    cursor.execute("""
                        INSERT INTO annotations
                        (id, image_id, label, class_name, score, bbox, polygon, area, manual)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        ann['id'],
                        new_image_id,
                        ann['label'],
                        ann['class_name'],
                        ann['score'],
                        json.dumps(ann['bbox']),
                        json.dumps(ann['polygon']),
                        ann['area'],
                        ann['manual']
                    ))

            # 更新项目
            cursor.execute("""
                UPDATE projects SET image_dir = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (image_dir, project_id))
        
        # 事务完成后，删除实际的图片文件
        for path in paths_to_delete:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                    print(f"[DB] 删除图片文件: {path}")
                except Exception as e:
                    print(f"[DB] 删除图片文件失败: {path}, 错误: {e}")

    def get_image(self, image_id: int) -> Optional[dict]:
        """获取单张图片（包含标注）"""
        with self._get_conn() as conn:
            cursor = conn.cursor()

            # 获取图片基本信息
            cursor.execute("SELECT * FROM images WHERE id = ?", (image_id,))
            row = cursor.fetchone()
            if not row:
                return None

            image = dict(row)
            image['annotated'] = bool(image['annotated'])

            # 获取标注
            cursor.execute("""
                SELECT id, label, class_name, score, bbox, polygon, area
                FROM annotations WHERE image_id = ?
            """, (image_id,))

            annotations = []
            for row in cursor.fetchall():
                ann = dict(row)
                ann['bbox'] = json.loads(row['bbox']) if row['bbox'] else []
                ann['polygon'] = json.loads(row['polygon']) if row['polygon'] else []
                annotations.append(ann)

            image['annotations'] = annotations
            return image

    def update_image_annotated(self, image_id: int, annotated: bool):
        """更新图片标注状态"""
        with self._get_conn() as conn:
            conn.execute("""
                UPDATE images SET annotated = ? WHERE id = ?
            """, (1 if annotated else 0, image_id))

    def mark_image_annotated(self, project_id: str, image_index: int, annotated: bool = True):
        """标记图片为已标注/未标注"""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE images SET annotated = ?
                WHERE project_id = ? AND index_in_project = ?
            """, (1 if annotated else 0, project_id, image_index))

    # ==================== 标注管理 ====================

    def add_annotations(self, project_id: str, image_index: int,
                        annotations: List[dict], label: str = None):
        """添加标注（增量保存，先删除指定类别的非手动标注）"""
        with self._get_conn() as conn:
            cursor = conn.cursor()

            # 获取image_id
            cursor.execute("""
                SELECT id FROM images
                WHERE project_id = ? AND index_in_project = ?
            """, (project_id, image_index))
            row = cursor.fetchone()
            if not row:
                raise ValueError(f"图片不存在: {image_index}")
            image_id = row['id']

            # 获取目标类别名称
            target_class = label
            if annotations and annotations[0].get('class_name'):
                target_class = annotations[0].get('class_name')

            # 先删除指定类别的非手动标注（保留手动标注）
            if target_class:
                cursor.execute("""
                    DELETE FROM annotations
                    WHERE image_id = ? AND class_name = ? AND (manual IS NULL OR manual = 0)
                """, (image_id, target_class))
                print(f"[DB] 删除类别 '{target_class}' 的 {cursor.rowcount} 个非手动标注")

            # 批量插入标注
            values = []
            for ann in annotations:
                ann_id = ann.get('id', str(uuid.uuid4())[:8])
                values.append((
                    ann_id,
                    image_id,
                    ann.get('label'),
                    ann.get('class_name') or label,
                    ann.get('score'),
                    json.dumps(ann.get('bbox')),
                    json.dumps(ann.get('polygon')),
                    ann.get('area'),
                    1 if ann.get('manual') else 0  # manual 标志
                ))

            cursor.executemany("""
                INSERT INTO annotations
                (id, image_id, label, class_name, score, bbox, polygon, area, manual)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, values)

            # 标记图片已标注
            self.update_image_annotated(image_id, True)

    def add_annotations_batch(self, annotations_data: List[tuple]):
        """批量添加标注（高性能版本，先删除指定类别的非手动标注）

        Args:
            annotations_data: [(project_id, image_index, annotations, label), ...]
        """
        with self._get_conn() as conn:
            cursor = conn.cursor()

            # 先批量删除指定类别的非手动标注
            for project_id, image_index, annotations, label in annotations_data:
                target_class = label
                if annotations and annotations[0].get('class_name'):
                    target_class = annotations[0].get('class_name')

                if target_class:
                    cursor.execute("""
                        SELECT id FROM images
                        WHERE project_id = ? AND index_in_project = ?
                    """, (project_id, image_index))
                    row = cursor.fetchone()
                    if row:
                        cursor.execute("""
                            DELETE FROM annotations
                            WHERE image_id = ? AND class_name = ? AND (manual IS NULL OR manual = 0)
                        """, (row['id'], target_class))
                        if cursor.rowcount > 0:
                            print(f"[DB] 批量删除类别 '{target_class}' 的 {cursor.rowcount} 个非手动标注")

            # 准备批量数据
            values = []
            image_ids_for_insert = set()
            all_image_ids = set()

            for project_id, image_index, annotations, label in annotations_data:
                # 获取image_id
                cursor.execute("""
                    SELECT id FROM images
                    WHERE project_id = ? AND index_in_project = ?
                """, (project_id, image_index))
                row = cursor.fetchone()
                if not row:
                    continue
                image_id = row['id']
                all_image_ids.add(image_id)

                # 添加标注
                if annotations:  # 只有当有标注时才添加
                    image_ids_for_insert.add(image_id)
                    for ann in annotations:
                        ann_id = ann.get('id', str(uuid.uuid4())[:8])
                        values.append((
                            ann_id,
                            image_id,
                            ann.get('label'),
                            ann.get('class_name') or label,
                            ann.get('score'),
                            json.dumps(ann.get('bbox')),
                            json.dumps(ann.get('polygon')),
                            ann.get('area'),
                            1 if ann.get('manual') else 0  # manual 标志
                        ))

            # 批量插入标注
            if values:
                cursor.executemany("""
                    INSERT INTO annotations
                    (id, image_id, label, class_name, score, bbox, polygon, area, manual)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, values)
                print(f"[DB] 批量插入 {len(values)} 个标注")

            # 批量更新图片标注状态
            # 需要检查每张图片是否还有任何标注
            for image_id in all_image_ids:
                cursor.execute("""
                    SELECT COUNT(*) as count FROM annotations WHERE image_id = ?
                """, (image_id,))
                count = cursor.fetchone()['count']
                cursor.execute("""
                    UPDATE images SET annotated = ? WHERE id = ?
                """, (1 if count > 0 else 0, image_id))

    def save_annotations(self, project_id: str, image_index: int, annotations: list):
        """保存标注（覆盖）"""
        with self._get_conn() as conn:
            cursor = conn.cursor()

            # 获取image_id
            cursor.execute("""
                SELECT id FROM images
                WHERE project_id = ? AND index_in_project = ?
            """, (project_id, image_index))
            row = cursor.fetchone()
            if not row:
                raise ValueError(f"图片不存在: {image_index}")
            image_id = row['id']

            # 删除旧标注
            cursor.execute("DELETE FROM annotations WHERE image_id = ?", (image_id,))

            # 添加新标注
            if annotations:
                values = []
                for ann in annotations:
                    ann_id = ann.get('id', str(uuid.uuid4())[:8])
                    values.append((
                        ann_id,
                        image_id,
                        ann.get('label'),
                        ann.get('class_name'),
                        ann.get('score'),
                        json.dumps(ann.get('bbox')),
                        json.dumps(ann.get('polygon')),
                        ann.get('area'),
                        ann.get('manual', 0)  # 支持 manual 字段，默认为0（AI标注）
                    ))

                cursor.executemany("""
                    INSERT INTO annotations
                    (id, image_id, label, class_name, score, bbox, polygon, area, manual)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, values)

            # 更新图片标注状态
            annotated = len(annotations) > 0
            self.update_image_annotated(image_id, annotated)

    def get_annotations(self, project_id: str, image_index: int) -> list:
        """获取标注"""
        with self._get_conn() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT a.id, a.label, a.class_name, a.score, a.bbox, a.polygon, a.area, a.manual
                FROM annotations a
                JOIN images i ON a.image_id = i.id
                WHERE i.project_id = ? AND i.index_in_project = ?
            """, (project_id, image_index))

            annotations = []
            for row in cursor.fetchall():
                ann = dict(row)
                ann['bbox'] = json.loads(row['bbox']) if row['bbox'] else []
                ann['polygon'] = json.loads(row['polygon']) if row['polygon'] else []
                # 将 manual 转换为布尔值
                if ann.get('manual') is not None:
                    ann['manual'] = bool(row['manual'])
                else:
                    ann['manual'] = False
                annotations.append(ann)

            return annotations

    def get_annotations_by_image_id(self, project_id: str, image_id: int) -> list:
        """根据图片ID获取标注"""
        with self._get_conn() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT a.id, a.label, a.class_name, a.score, a.bbox, a.polygon, a.area, a.manual
                FROM annotations a
                JOIN images i ON a.image_id = i.id
                WHERE i.project_id = ? AND i.id = ?
            """, (project_id, image_id))

            annotations = []
            for row in cursor.fetchall():
                ann = dict(row)
                ann['bbox'] = json.loads(row['bbox']) if row['bbox'] else []
                ann['polygon'] = json.loads(row['polygon']) if row['polygon'] else []
                # 将 manual 转换为布尔值
                if ann.get('manual') is not None:
                    ann['manual'] = bool(row['manual'])
                else:
                    ann['manual'] = False
                annotations.append(ann)

            return annotations

    def update_annotation(self, project_id: str, image_index: int,
                          annotation_id: str, updates: dict):
        """更新单个标注"""
        with self._get_conn() as conn:
            cursor = conn.cursor()

            # 获取image_id
            cursor.execute("""
                SELECT id FROM images
                WHERE project_id = ? AND index_in_project = ?
            """, (project_id, image_index))
            row = cursor.fetchone()
            if not row:
                raise ValueError(f"图片不存在: {image_index}")
            image_id = row['id']

            # 构建更新语句
            update_fields = []
            update_values = []

            for key, value in updates.items():
                if key in ['label', 'class_name', 'score', 'area', 'manual']:
                    update_fields.append(f"{key} = ?")
                    update_values.append(value)
                elif key == 'bbox' or key == 'polygon':
                    update_fields.append(f"{key} = ?")
                    update_values.append(json.dumps(value))

            if update_fields:
                update_values.extend([annotation_id, image_id])
                cursor.execute(f"""
                    UPDATE annotations SET {', '.join(update_fields)}
                    WHERE id = ? AND image_id = ?
                """, update_values)

    def delete_annotation(self, project_id: str, image_index: int, annotation_id: str):
        """删除单个标注"""
        with self._get_conn() as conn:
            cursor = conn.cursor()

            # 获取image_id
            cursor.execute("""
                SELECT id, (SELECT COUNT(*) FROM annotations WHERE image_id = i.id) as ann_count
                FROM images i
                WHERE project_id = ? AND index_in_project = ?
            """, (project_id, image_index))
            row = cursor.fetchone()
            if not row:
                raise ValueError(f"图片不存在: {image_index}")
            image_id = row['id']

            # 删除标注
            cursor.execute("""
                DELETE FROM annotations WHERE id = ? AND image_id = ?
            """, (annotation_id, image_id))

            # 检查是否还有标注
            cursor.execute("""
                SELECT COUNT(*) as count FROM annotations WHERE image_id = ?
            """, (image_id,))
            new_count = cursor.fetchone()['count']

            # 更新图片标注状态
            self.update_image_annotated(image_id, new_count > 0)

    def delete_class_annotations(self, project_id: str, class_name: str) -> int:
        """删除项目中某个类别的所有标注
        
        Args:
            project_id: 项目ID
            class_name: 类别名称
            
        Returns:
            删除的标注数量
        """
        with self._get_conn() as conn:
            cursor = conn.cursor()
            
            # 删除指定类别的所有标注
            cursor.execute("""
                DELETE FROM annotations
                WHERE image_id IN (
                    SELECT id FROM images WHERE project_id = ?
                ) AND class_name = ?
            """, (project_id, class_name))
            
            deleted_count = cursor.rowcount
            
            # 更新所有图片的标注状态
            cursor.execute("""
                SELECT id FROM images WHERE project_id = ?
            """, (project_id,))
            
            for row in cursor.fetchall():
                image_id = row['id']
                cursor.execute("""
                    SELECT COUNT(*) as count FROM annotations WHERE image_id = ?
                """, (image_id,))
                has_annotations = cursor.fetchone()['count'] > 0
                self.update_image_annotated(image_id, has_annotations)
            
            return deleted_count

    def delete_image(self, project_id: str, image_index: int) -> bool:
        """删除图片及其所有标注
        
        Args:
            project_id: 项目ID
            image_index: 图片在项目中的索引
            
        Returns:
            是否成功删除
        """
        import os
        
        with self._get_conn() as conn:
            cursor = conn.cursor()
            
            # 获取要删除的图片ID和路径
            cursor.execute("""
                SELECT id, filename, path FROM images
                WHERE project_id = ? AND index_in_project = ?
            """, (project_id, image_index))
            row = cursor.fetchone()
            
            if not row:
                return False
            
            image_id = row['id']
            filename = row['filename']
            image_path = row['path']
            
            # 删除该图片的所有标注
            cursor.execute("DELETE FROM annotations WHERE image_id = ?", (image_id,))
            
            # 删除图片记录
            cursor.execute("DELETE FROM images WHERE id = ?", (image_id,))
            
            # 更新后续图片的索引
            cursor.execute("""
                UPDATE images
                SET index_in_project = index_in_project - 1
                WHERE project_id = ? AND index_in_project > ?
            """, (project_id, image_index))
            
            print(f"[DB] 删除图片记录: {filename}, 索引 {image_index}")
            
        # 删除实际的图片文件
        if image_path and os.path.exists(image_path):
            try:
                os.remove(image_path)
                print(f"[DB] 删除图片文件: {image_path}")
            except Exception as e:
                print(f"[DB] 删除图片文件失败: {image_path}, 错误: {e}")
        
        return True

    def clear_non_manual_annotations(self, project_id: str) -> int:
        """清理项目中所有非手动标注（AI自动生成的标注）
        
        Args:
            project_id: 项目ID
            
        Returns:
            删除的标注数量
        """
        with self._get_conn() as conn:
            cursor = conn.cursor()
            
            # 删除所有非手动标注（manual = 0 或 NULL）
            cursor.execute("""
                DELETE FROM annotations
                WHERE image_id IN (
                    SELECT id FROM images WHERE project_id = ?
                ) AND (manual IS NULL OR manual = 0)
            """, (project_id,))
            
            deleted_count = cursor.rowcount
            print(f"[DB] 清理项目 {project_id} 的非手动标注: 删除 {deleted_count} 个")
            
            # 更新所有图片的标注状态
            cursor.execute("""
                SELECT id FROM images WHERE project_id = ?
            """, (project_id,))
            
            for row in cursor.fetchall():
                image_id = row['id']
                cursor.execute("""
                    SELECT COUNT(*) as count FROM annotations WHERE image_id = ?
                """, (image_id,))
                has_annotations = cursor.fetchone()['count'] > 0
                self.update_image_annotated(image_id, has_annotations)
            
            return deleted_count

    def get_annotations_by_class(self, project_id: str, class_name: str, manual_only: bool = False, summary_only: bool = True) -> List[Dict]:
        """获取指定类别的标注信息
        
        Args:
            project_id: 项目ID
            class_name: 类别名称
            manual_only: 是否只获取手动标注（默认False，获取所有标注）
            summary_only: 是否只返回概况信息（默认True，减少数据传输）
            
        Returns:
            标注列表或概况统计
        """
        with self._get_conn() as conn:
            cursor = conn.cursor()
            
            if summary_only:
                # 只返回按图片分组的概况信息
                if manual_only:
                    cursor.execute("""
                        SELECT 
                            i.index_in_project,
                            i.filename,
                            COUNT(*) as annotation_count,
                            COUNT(CASE WHEN a.manual = 1 THEN 1 END) as manual_count,
                            COUNT(CASE WHEN a.manual IS NULL OR a.manual = 0 THEN 1 END) as ai_count,
                            AVG(CASE WHEN a.manual IS NULL OR a.manual = 0 THEN a.score END) as avg_ai_score,
                            MIN(CASE WHEN a.manual IS NULL OR a.manual = 0 THEN a.score END) as min_ai_score,
                            MAX(CASE WHEN a.manual IS NULL OR a.manual = 0 THEN a.score END) as max_ai_score,
                            COUNT(CASE WHEN (a.manual IS NULL OR a.manual = 0) AND (a.score IS NULL OR a.score < 0.45) THEN 1 END) as low_confidence_count
                        FROM annotations a
                        JOIN images i ON a.image_id = i.id
                        WHERE i.project_id = ? 
                            AND a.class_name = ?
                            AND a.manual = 1
                        GROUP BY i.id, i.index_in_project, i.filename
                        ORDER BY i.index_in_project
                    """, (project_id, class_name))
                else:
                    cursor.execute("""
                        SELECT 
                            i.index_in_project,
                            i.filename,
                            COUNT(*) as annotation_count,
                            COUNT(CASE WHEN a.manual = 1 THEN 1 END) as manual_count,
                            COUNT(CASE WHEN a.manual IS NULL OR a.manual = 0 THEN 1 END) as ai_count,
                            AVG(CASE WHEN a.manual IS NULL OR a.manual = 0 THEN a.score END) as avg_ai_score,
                            MIN(CASE WHEN a.manual IS NULL OR a.manual = 0 THEN a.score END) as min_ai_score,
                            MAX(CASE WHEN a.manual IS NULL OR a.manual = 0 THEN a.score END) as max_ai_score,
                            COUNT(CASE WHEN (a.manual IS NULL OR a.manual = 0) AND (a.score IS NULL OR a.score < 0.45) THEN 1 END) as low_confidence_count
                        FROM annotations a
                        JOIN images i ON a.image_id = i.id
                        WHERE i.project_id = ? 
                            AND a.class_name = ?
                        GROUP BY i.id, i.index_in_project, i.filename
                        ORDER BY i.index_in_project
                    """, (project_id, class_name))
                
                rows = cursor.fetchall()
                summary = []
                for row in rows:
                    summary.append({
                        'image_index': row['index_in_project'],
                        'filename': row['filename'],
                        'annotation_count': row['annotation_count'],
                        'manual_count': row['manual_count'],
                        'ai_count': row['ai_count'],
                        'avg_ai_score': round(row['avg_ai_score'], 4) if row['avg_ai_score'] else None,
                        'min_ai_score': round(row['min_ai_score'], 4) if row['min_ai_score'] else None,
                        'max_ai_score': round(row['max_ai_score'], 4) if row['max_ai_score'] else None,
                        'low_confidence_count': row['low_confidence_count']
                    })
                
                print(f"[DB] 获取项目 {project_id} 类别 {class_name} 的标注概况: {len(summary)} 张图片")
                return summary
            
            else:
                # 返回详细标注信息（用于需要时）
                cursor.execute("""
                    SELECT 
                        i.id as image_id,
                        i.filename,
                        i.index_in_project,
                        a.id as annotation_id,
                        a.label,
                        a.class_name,
                        a.score,
                        a.bbox,
                        a.polygon,
                        a.area,
                        a.manual,
                        a.created_at
                    FROM annotations a
                    JOIN images i ON a.image_id = i.id
                    WHERE i.project_id = ? 
                        AND a.class_name = ?
                    ORDER BY i.index_in_project, a.created_at
                """, (project_id, class_name))
                
                rows = cursor.fetchall()
                annotations = []
                for row in rows:
                    annotations.append({
                        'image_id': row['image_id'],
                        'filename': row['filename'],
                        'image_index': row['index_in_project'],
                        'annotation_id': row['annotation_id'],
                        'label': row['label'],
                        'class_name': row['class_name'],
                        'score': row['score'],
                        'bbox': json.loads(row['bbox']) if row['bbox'] else None,
                        'polygon': json.loads(row['polygon']) if row['polygon'] else None,
                        'area': row['area'],
                        'manual': row['manual'],
                        'created_at': row['created_at']
                    })
                
                print(f"[DB] 获取项目 {project_id} 类别 {class_name} 的详细标注: 找到 {len(annotations)} 个")
                return annotations

    def clear_low_confidence_annotations(self, project_id: str, confidence_threshold: float = 0.45, class_name: str = None) -> int:
        """清理置信度低于指定值的标注
        
        Args:
            project_id: 项目ID
            confidence_threshold: 置信度阈值（默认0.45）
            class_name: 可选，指定类别，如果为None则清理所有类别
            
        Returns:
            删除的标注数量
        """
        with self._get_conn() as conn:
            cursor = conn.cursor()
            
            # 查询需要删除的标注（非手动且置信度低于阈值）
            if class_name:
                cursor.execute("""
                    DELETE FROM annotations
                    WHERE image_id IN (
                        SELECT id FROM images WHERE project_id = ?
                    ) AND (manual IS NULL OR manual = 0)
                    AND (score IS NULL OR score < ?)
                    AND class_name = ?
                """, (project_id, confidence_threshold, class_name))
            else:
                cursor.execute("""
                    DELETE FROM annotations
                    WHERE image_id IN (
                        SELECT id FROM images WHERE project_id = ?
                    ) AND (manual IS NULL OR manual = 0)
                    AND (score IS NULL OR score < ?)
                """, (project_id, confidence_threshold))
            
            deleted_count = cursor.rowcount
            class_info = f"类别 {class_name} 的" if class_name else "所有"
            print(f"[DB] 清理项目 {project_id} {class_info}低置信度标注（<{confidence_threshold}）: 删除 {deleted_count} 个")
            
            # 更新所有图片的标注状态
            cursor.execute("""
                SELECT id FROM images WHERE project_id = ?
            """, (project_id,))
            
            for row in cursor.fetchall():
                image_id = row['id']
                cursor.execute("""
                    SELECT COUNT(*) as count FROM annotations WHERE image_id = ?
                """, (image_id,))
                has_annotations = cursor.fetchone()['count'] > 0
                self.update_image_annotated(image_id, has_annotations)
            
            return deleted_count

    def get_annotation_stats_by_class(self, project_id: str) -> List[Dict]:
        """获取各类别的标注统计信息
        
        Args:
            project_id: 项目ID
            
        Returns:
            统计信息列表，每个元素包含类别名称、总数、手动标注数、AI标注数、平均置信度
        """
        with self._get_conn() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT 
                    class_name,
                    COUNT(*) as total_count,
                    SUM(CASE WHEN manual = 1 THEN 1 ELSE 0 END) as manual_count,
                    SUM(CASE WHEN manual IS NULL OR manual = 0 THEN 1 ELSE 0 END) as ai_count,
                    AVG(score) as avg_score,
                    MIN(score) as min_score,
                    MAX(score) as max_score
                FROM annotations a
                JOIN images i ON a.image_id = i.id
                WHERE i.project_id = ?
                GROUP BY class_name
                ORDER BY total_count DESC
            """, (project_id,))
            
            rows = cursor.fetchall()
            stats = []
            for row in rows:
                stats.append({
                    'class_name': row['class_name'],
                    'total_count': row['total_count'],
                    'manual_count': row['manual_count'],
                    'ai_count': row['ai_count'],
                    'avg_score': round(row['avg_score'], 4) if row['avg_score'] else None,
                    'min_score': round(row['min_score'], 4) if row['min_score'] else None,
                    'max_score': round(row['max_score'], 4) if row['max_score'] else None
                })
            
            return stats

    # ==================== 统计信息 ====================

    def get_annotation_stats(self, project_id: str) -> dict:
        """获取标注统计"""
        with self._get_conn() as conn:
            cursor = conn.cursor()

            # 图片统计
            cursor.execute("""
                SELECT
                    COUNT(*) as total,
                    SUM(annotated) as annotated
                FROM images WHERE project_id = ?
            """, (project_id,))
            row = cursor.fetchone()

            total = row['total'] or 0
            annotated = row['annotated'] or 0

            # 标注统计
            cursor.execute("""
                SELECT COUNT(*) as count
                FROM annotations a
                JOIN images i ON a.image_id = i.id
                WHERE i.project_id = ?
            """, (project_id,))
            total_annotations = cursor.fetchone()['count'] or 0

            return {
                'total_images': total,
                'annotated_images': annotated,
                'unannotated_images': total - annotated,
                'total_annotations': total_annotations,
                'progress': annotated / total * 100 if total > 0 else 0
            }

    def get_db_stats(self) -> dict:
        """获取数据库统计信息"""
        import os

        with self._get_conn() as conn:
            cursor = conn.cursor()

            stats = {}

            # 项目数量
            cursor.execute("SELECT COUNT(*) as count FROM projects")
            stats['projects'] = cursor.fetchone()['count']

            # 图片数量
            cursor.execute("SELECT COUNT(*) as count FROM images")
            stats['images'] = cursor.fetchone()['count']

            # 标注数量
            cursor.execute("SELECT COUNT(*) as count FROM annotations")
            stats['annotations'] = cursor.fetchone()['count']

            # 数据库文件大小
            if self.db_path.exists():
                stats['db_size_mb'] = round(os.path.getsize(self.db_path) / 1024 / 1024, 2)

            return stats

    def close(self):
        """关闭数据库连接"""
        if hasattr(self._local, 'conn') and self._local.conn:
            self._local.conn.close()
            self._local.conn = None
