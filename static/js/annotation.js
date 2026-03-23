/**
 * SAM3 数据标注工具 - 前端交互逻辑
 */

// 全局状态
const state = {
    projectId: null,
    images: [],
    currentIndex: 0,
    annotations: [],
    classes: [],
    currentClass: null,  // 当前选中的类名
    currentTool: 'point',
    isPositive: true,
    confidence: 0.5,  // 与配置文件保持一致
    zoom: 1,
    pan: { x: 0, y: 0 },
    drawing: false,
    drawStart: null,
    panning: false,      // 平移拖动中
    panStart: null,      // 拖动起始位置
    spacePressed: false, // 空格键按下状态
    selectedAnnotation: null,
    tempPoints: [],
    tempBoxes: [],  // 临时框数组，每个框包含 {x1, y1, x2, y2, label}
    tempPolygon: [], // 手动绘制的多边形顶点
    nmsConfig: {     // NMS配置
        enabled: true,
        iouThreshold: 0.4,
        overlapMode: 'iou',
        minAreaRatio: 0.1,
        maskLevel: true
    },
    filterConfidence: 0.45,  // 低置信度筛选阈值
    filteredImages: []  // 筛选后的图片索引列表
};

// Canvas相关
let canvas, ctx;
let currentImage = null;

// 性能优化：缓存和帧控制
let staticCanvas = null;  // 离屏canvas缓存静态内容
let staticCtx = null;
let staticCacheDirty = true;  // 静态缓存是否需要更新
let rafId = null;  // requestAnimationFrame ID
let pendingDraw = null;  // 待绘制的动态内容

// 颜色映射
const colors = [
    '#e94560', '#4dabf7', '#4ade80', '#fbbf24', '#a78bfa',
    '#f472b6', '#22d3d8', '#fb923c', '#84cc16', '#6366f1'
];

// 初始化
document.addEventListener('DOMContentLoaded', () => {
    initCanvas();
    initEventListeners();
    loadProjects();
    restoreWorkState();  // 恢复上次工作状态
    restorePanelState(); // 恢复面板折叠状态
    handleResponsiveCollapse(); // 初始化响应式
    initBatchModeListener(); // 初始化批量处理模式监听
});

// 批量处理模式切换监听
function initBatchModeListener() {
    const batchModeSelect = document.getElementById('batchMode');
    const batchModeHint = document.getElementById('batchModeHint');

    if (batchModeSelect && batchModeHint) {
        const updateHint = () => {
            const mode = batchModeSelect.value;
            const maxWorkersDiv = document.getElementById('maxWorkersDiv');
            if (mode === 'concurrent') {
                batchModeHint.innerHTML = '<i class="bi bi-info-circle"></i> 并发模式：多线程 + 特征缓存 + GPU加速，<i class="bi bi-bullseye"></i> 置信度: <span id="batchConfidenceValue">0.50</span>';
                maxWorkersDiv.style.display = 'block';
            } else {
                batchModeHint.innerHTML = '<i class="bi bi-exclamation-triangle"></i> 串行模式：单线程处理，速度较慢';
                maxWorkersDiv.style.display = 'none';
            }
        };

        batchModeSelect.addEventListener('change', updateHint);
        updateHint(); // 初始化提示
    }
}

// 保存工作状态到localStorage
function saveWorkState() {
    if (state.projectId) {
        const workState = {
            projectId: state.projectId,
            currentIndex: state.currentIndex,
            timestamp: Date.now()
        };
        localStorage.setItem('sam3_work_state', JSON.stringify(workState));
    }
}

// 恢复工作状态
async function restoreWorkState() {
    const saved = localStorage.getItem('sam3_work_state');
    if (!saved) return;

    try {
        const workState = JSON.parse(saved);

        // 恢复项目
        if (workState.projectId) {
            const success = await selectProject(workState.projectId);

            if (success) {
                // 自动扫描文件夹更新图片列表
                await rescanProjectImages();

                // 恢复图片位置
                if (workState.currentIndex >= 0 && workState.currentIndex < state.images.length) {
                    loadImage(workState.currentIndex);
                } else if (state.images.length > 0) {
                    loadImage(0);
                }
            }
        }
    } catch (e) {
        console.error('恢复工作状态失败:', e);
    }
}

// 重新扫描项目图片文件夹
async function rescanProjectImages() {
    if (!state.projectId) return;

    try {
        // 使用轻量级API获取项目基本信息
        const response = await fetch(`/api/project/${state.projectId}/info`);
        if (!response.ok) {
            console.error('获取项目信息失败:', response.status, response.statusText);
            return;
        }

        const data = await response.json();
        if (!data.success) {
            console.error('获取项目信息失败:', data.error);
            return;
        }

        if (data.project && data.project.image_dir) {
            const scanResponse = await fetch(`/api/project/${state.projectId}/load_images`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image_dir: data.project.image_dir })
            });

            if (!scanResponse.ok) {
                console.error('扫描图片失败:', scanResponse.status, scanResponse.statusText);
                const errorText = await scanResponse.text();
                console.error('错误详情:', errorText);
                return;
            }

            const scanData = await scanResponse.json();
            if (scanData.success) {
                state.images = scanData.images;
                updateImageList();
                showToast('成功', `扫描完成，共 ${scanData.count} 张图片`);
            } else {
                console.error('扫描图片失败:', scanData.error);
                showToast('错误', scanData.error || '扫描图片失败', 'danger');
            }
        }
    } catch (e) {
        console.error('扫描文件夹失败:', e);
        showToast('错误', '扫描文件夹失败', 'danger');
    }
}

// 清除当前图片的所有标注
async function clearCurrentAnnotations() {
    if (state.annotations.length === 0) {
        showToast('提示', '当前没有标注');
        return;
    }

    if (confirm(`确定要清除当前图片的 ${state.annotations.length} 个标注吗？`)) {
        state.annotations = [];
        state.selectedAnnotation = null;
        invalidateStaticCache();  // 标注变化，更新缓存
        updateAnnotationList();
        redraw();
        // 自动保存（不标记为手动）
        await saveAnnotations(false, false);
        showToast('成功', '已清除并保存');
    }
}

function initCanvas() {
    canvas = document.getElementById('annotationCanvas');
    ctx = canvas.getContext('2d');

    canvas.addEventListener('mousedown', onMouseDown);
    canvas.addEventListener('mousemove', onMouseMove);
    canvas.addEventListener('mouseup', onMouseUp);
    canvas.addEventListener('wheel', onWheel);
    canvas.addEventListener('dblclick', onDoubleClick);

    // 禁用右键菜单（用于右键拖动）
    canvas.addEventListener('contextmenu', onContextMenu);

    // 鼠标离开canvas时结束拖动
    canvas.addEventListener('mouseleave', () => {
        if (state.panning) {
            state.panning = false;
            state.panStart = null;
            canvas.style.cursor = state.currentTool === 'edit' ? 'default' : 'crosshair';
        }
    });

    // 键盘快捷键
    document.addEventListener('keydown', onKeyDown);
    document.addEventListener('keyup', onKeyUp);

    // 初始化工具栏按钮状态
    initToolbarState();
}

function initToolbarState() {
    // 设置默认工具为 point
    const pointBtn = document.getElementById('toolPoint');
    if (pointBtn) pointBtn.classList.add('active');

    // 设置默认为正样本
    const positiveBtn = document.getElementById('labelPositive');
    if (positiveBtn) positiveBtn.classList.add('active');
}

function initEventListeners() {
    // 置信度滑块
    const confidenceSlider = document.getElementById('confidenceSlider');
    if (confidenceSlider) {
        confidenceSlider.addEventListener('input', (e) => {
            state.confidence = e.target.value / 100;
            const confidenceValue = document.getElementById('confidenceValue');
            if (confidenceValue) confidenceValue.textContent = state.confidence.toFixed(2);
            // 更新批量分割的置信度显示
            const batchConfidenceValue = document.getElementById('batchConfidenceValue');
            if (batchConfidenceValue) batchConfidenceValue.textContent = state.confidence.toFixed(2);
        });
    }

    // 清理置信度滑块
    const clearConfidenceSlider = document.getElementById('clearConfidenceSlider');
    if (clearConfidenceSlider) {
        clearConfidenceSlider.addEventListener('input', (e) => {
            const clearConfidenceValue = document.getElementById('clearConfidenceValue');
            if (clearConfidenceValue) clearConfidenceValue.textContent = (e.target.value / 100).toFixed(2);
        });
    }

    // 筛选置信度滑块
    const filterConfidenceSlider = document.getElementById('filterConfidenceSlider');
    if (filterConfidenceSlider) {
        filterConfidenceSlider.addEventListener('input', (e) => {
            state.filterConfidence = e.target.value / 100;
            const filterConfidenceValue = document.getElementById('filterConfidenceValue');
            if (filterConfidenceValue) filterConfidenceValue.textContent = state.filterConfidence.toFixed(2);
        });
    }

    // 图片搜索
    const imageSearch = document.getElementById('imageSearch');
    if (imageSearch) {
        imageSearch.addEventListener('input', (e) => {
            filterImages(e.target.value);
        });
    }

    // 文本提示回车
    const textPrompt = document.getElementById('textPrompt');
    if (textPrompt) {
        textPrompt.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') segmentByText();
        });
    }

    // NMS事件监听
    const enableNms = document.getElementById('enableNms');
    if (enableNms) {
        enableNms.addEventListener('change', (e) => {
            state.nmsConfig.enabled = e.target.checked;
        });

        const nmsIouSlider = document.getElementById('nmsIouSlider');
        if (nmsIouSlider) {
            nmsIouSlider.addEventListener('input', (e) => {
                state.nmsConfig.iouThreshold = e.target.value / 100;
                const nmsIouValue = document.getElementById('nmsIouValue');
                if (nmsIouValue) nmsIouValue.textContent = state.nmsConfig.iouThreshold.toFixed(2);
            });
        }

        const nmsOverlapMode = document.getElementById('nmsOverlapMode');
        if (nmsOverlapMode) {
            nmsOverlapMode.addEventListener('change', (e) => {
                state.nmsConfig.overlapMode = e.target.value;
            });
        }

        const nmsMinAreaSlider = document.getElementById('nmsMinAreaSlider');
        if (nmsMinAreaSlider) {
            nmsMinAreaSlider.addEventListener('input', (e) => {
                state.nmsConfig.minAreaRatio = e.target.value / 100;
                const nmsMinAreaValue = document.getElementById('nmsMinAreaValue');
                if (nmsMinAreaValue) nmsMinAreaValue.textContent = state.nmsConfig.minAreaRatio.toFixed(2);
            });
        }

        const nmsMaskLevel = document.getElementById('nmsMaskLevel');
        if (nmsMaskLevel) {
            nmsMaskLevel.addEventListener('change', (e) => {
                state.nmsConfig.maskLevel = e.target.checked;
            });
        }
    }
}

// ==================== 工具切换 ====================

function setTool(tool) {
    state.currentTool = tool;
    state.tempPoints = [];
    state.tempBoxes = [];
    state.tempPolygon = [];  // 清除临时多边形

    // 移除所有工具按钮的 active 状态
    document.querySelectorAll('.toolbar-group .btn-tool').forEach(btn => {
        // 只处理工具按钮（不处理正负样本按钮）
        if (btn.id && btn.id.startsWith('tool')) {
            btn.classList.remove('active');
        }
    });

    // 激活当前工具按钮
    const toolBtn = document.getElementById('tool' + tool.charAt(0).toUpperCase() + tool.slice(1));
    if (toolBtn) {
        toolBtn.classList.add('active');
    }

    // 更新光标
    if (tool === 'edit') {
        canvas.style.cursor = 'default';
    } else {
        canvas.style.cursor = 'crosshair';
    }

    redraw();
}

function setLabel(isPositive) {
    state.isPositive = isPositive;
    document.getElementById('labelPositive').classList.toggle('active', isPositive);
    document.getElementById('labelNegative').classList.toggle('active', !isPositive);
}

// ==================== Canvas事件处理 ====================

// 获取鼠标在Canvas上的真实坐标
function getCanvasCoords(e) {
    const rect = canvas.getBoundingClientRect();
    // 计算Canvas的缩放比例（CSS尺寸 vs 实际尺寸）
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    // 计算鼠标在Canvas上的实际坐标
    const x = (e.clientX - rect.left) * scaleX / state.zoom;
    const y = (e.clientY - rect.top) * scaleY / state.zoom;
    return { x, y };
}

function onMouseDown(e) {
    e.preventDefault();

    // 右键长按拖动视图
    if (e.button === 2) {
        state.panning = true;
        state.panStart = { x: e.clientX, y: e.clientY };
        canvas.style.cursor = 'grabbing';
        return;
    }

    // 中键拖动 或 空格+左键拖动（备用方式）
    if (e.button === 1 || (e.button === 0 && state.spacePressed)) {
        state.panning = true;
        state.panStart = { x: e.clientX, y: e.clientY };
        canvas.style.cursor = 'grabbing';
        return;
    }

    // 左键操作
    if (e.button !== 0) return;

    const { x, y } = getCanvasCoords(e);

    if (state.currentTool === 'point') {
        addPoint(x, y);
    } else if (state.currentTool === 'box') {
        state.drawing = true;
        state.drawStart = { x, y };
    } else if (state.currentTool === 'edit') {
        selectAnnotationAt(x, y);
    } else if (state.currentTool === 'polygon') {
        addPolygonPoint(x, y);
    }
}

// 缓存的鼠标位置，用于节流
let lastMouseX = 0, lastMouseY = 0;

function onMouseMove(e) {
    // 平移拖动中
    if (state.panning && state.panStart) {
        const container = document.getElementById('canvasContainer');
        const dx = e.clientX - state.panStart.x;
        const dy = e.clientY - state.panStart.y;

        container.scrollLeft -= dx;
        container.scrollTop -= dy;

        state.panStart = { x: e.clientX, y: e.clientY };
        return;
    }

    // 框选绘制中
    if (!state.drawing) return;

    // 节流：如果鼠标移动距离太小，跳过绘制
    const moveDist = Math.abs(e.clientX - lastMouseX) + Math.abs(e.clientY - lastMouseY);
    if (moveDist < 2) return;  // 移动小于2像素时跳过
    lastMouseX = e.clientX;
    lastMouseY = e.clientY;

    const { x, y } = getCanvasCoords(e);

    if (state.currentTool === 'box' && state.drawStart) {
        // 使用 requestAnimationFrame 优化性能
        if (rafId) {
            cancelAnimationFrame(rafId);
        }
        const dynamicBox = {
            x1: Math.min(state.drawStart.x, x),
            y1: Math.min(state.drawStart.y, y),
            x2: Math.max(state.drawStart.x, x),
            y2: Math.max(state.drawStart.y, y)
        };
        rafId = requestAnimationFrame(() => {
            quickRedraw(dynamicBox);
            rafId = null;
        });
    }
}

function onMouseUp(e) {
    // 结束平移拖动
    if (state.panning) {
        state.panning = false;
        state.panStart = null;
        updateCursor();
        return;
    }

    if (!state.drawing) return;

    const { x, y } = getCanvasCoords(e);

    if (state.currentTool === 'box' && state.drawStart) {
        const box = normalizeBox(state.drawStart.x, state.drawStart.y, x, y);
        if (box.width > 5 && box.height > 5) {
            // 保存临时框，包含当前的正负样本标签
            box.label = state.isPositive;
            state.tempBoxes.push(box);
            redraw();
            const labelStr = state.isPositive ? '正样本' : '负样本';
            showToast('提示', `${labelStr}框已添加，可继续添加或点击"分割"按钮`);
        }
    }

    state.drawing = false;
    state.drawStart = null;
}

function onDoubleClick(e) {
    // 双击完成多边形绘制
    if (state.currentTool === 'polygon' && state.tempPolygon.length >= 3) {
        e.preventDefault();
        finishPolygon();
    }
}

// 禁用右键菜单
function onContextMenu(e) {
    e.preventDefault();
    return false;
}

// 缩放节流控制
let zoomRafId = null;
let pendingZoomData = null;

// 滚轮缩放 - 以鼠标位置为中心
function onWheel(e) {
    e.preventDefault();
    if (!currentImage) return;

    const container = document.getElementById('canvasContainer');
    const wrapper = document.getElementById('canvasWrapper');
    const canvasRect = canvas.getBoundingClientRect();
    const containerRect = container.getBoundingClientRect();

    // 鼠标在canvas上的位置
    const mouseXOnCanvas = e.clientX - canvasRect.left;
    const mouseYOnCanvas = e.clientY - canvasRect.top;

    // 鼠标指向的图像坐标（原始图像上的位置）
    const imgX = mouseXOnCanvas / state.zoom;
    const imgY = mouseYOnCanvas / state.zoom;

    // 鼠标在容器视口中的位置
    const mouseXInViewport = e.clientX - containerRect.left;
    const mouseYInViewport = e.clientY - containerRect.top;

    // 计算新的缩放比例
    const oldZoom = state.zoom;
    const delta = e.deltaY > 0 ? 0.9 : 1.1;
    const newZoom = Math.max(0.1, Math.min(10, oldZoom * delta));

    if (newZoom === oldZoom) return;

    // 使用 requestAnimationFrame 节流缩放操作
    pendingZoomData = {
        newZoom,
        imgX,
        imgY,
        mouseXInViewport,
        mouseYInViewport,
        container,
        wrapper
    };

    if (!zoomRafId) {
        zoomRafId = requestAnimationFrame(() => {
            if (pendingZoomData) {
                const data = pendingZoomData;
                state.zoom = data.newZoom;
                invalidateStaticCache();  // 缩放变化，更新缓存
                redraw();

                // 获取wrapper的padding（使用计算样式）
                const wrapperStyle = window.getComputedStyle(data.wrapper);
                const paddingLeft = parseFloat(wrapperStyle.paddingLeft);
                const paddingTop = parseFloat(wrapperStyle.paddingTop);

                // 缩放后，该图像点在新canvas上的位置
                const newPointOnCanvasX = data.imgX * data.newZoom;
                const newPointOnCanvasY = data.imgY * data.newZoom;

                // 该点在wrapper中的位置 = padding + canvas上的位置
                const pointInWrapperX = paddingLeft + newPointOnCanvasX;
                const pointInWrapperY = paddingTop + newPointOnCanvasY;

                // 设置滚动位置，使该点保持在鼠标位置
                data.container.scrollLeft = pointInWrapperX - data.mouseXInViewport;
                data.container.scrollTop = pointInWrapperY - data.mouseYInViewport;

                // 更新缩放显示
                updateZoomDisplay();
                pendingZoomData = null;
            }
            zoomRafId = null;
        });
    }
}

// ==================== 面板折叠功能 ====================

// 切换左侧面板
function toggleLeftPanel() {
    const panel = document.getElementById('leftPanel');
    const expandBtn = document.getElementById('leftExpandBtn');

    // 如果有auto-collapse，先移除它
    const wasAutoCollapsed = panel.classList.contains('auto-collapse');
    if (wasAutoCollapsed) {
        panel.classList.remove('auto-collapse');
        expandBtn.style.display = 'none';
        return; // 只是取消自动折叠，不改变手动状态
    }

    panel.classList.toggle('collapsed');

    // 控制展开按钮显示
    if (panel.classList.contains('collapsed')) {
        expandBtn.style.display = 'flex';
    } else {
        expandBtn.style.display = 'none';
    }

    // 保存状态
    localStorage.setItem('leftPanelCollapsed', panel.classList.contains('collapsed'));
}

// 切换右侧面板
function toggleRightPanel() {
    const panel = document.getElementById('rightPanel');
    const expandBtn = document.getElementById('rightExpandBtn');

    // 如果有auto-collapse，先移除它
    const wasAutoCollapsed = panel.classList.contains('auto-collapse');
    if (wasAutoCollapsed) {
        panel.classList.remove('auto-collapse');
        expandBtn.style.display = 'none';
        return; // 只是取消自动折叠，不改变手动状态
    }

    panel.classList.toggle('collapsed');

    // 控制展开按钮显示
    if (panel.classList.contains('collapsed')) {
        expandBtn.style.display = 'flex';
    } else {
        expandBtn.style.display = 'none';
    }

    // 保存状态
    localStorage.setItem('rightPanelCollapsed', panel.classList.contains('collapsed'));
}

// 恢复面板状态
function restorePanelState() {
    const leftCollapsed = localStorage.getItem('leftPanelCollapsed') === 'true';
    const rightCollapsed = localStorage.getItem('rightPanelCollapsed') === 'true';

    if (leftCollapsed) {
        document.getElementById('leftPanel').classList.add('collapsed');
        document.getElementById('leftExpandBtn').style.display = 'flex';
    }

    if (rightCollapsed) {
        document.getElementById('rightPanel').classList.add('collapsed');
        document.getElementById('rightExpandBtn').style.display = 'flex';
    }
}

// 响应式自动折叠
function handleResponsiveCollapse() {
    const width = window.innerWidth;
    const leftPanel = document.getElementById('leftPanel');
    const rightPanel = document.getElementById('rightPanel');
    const leftExpandBtn = document.getElementById('leftExpandBtn');
    const rightExpandBtn = document.getElementById('rightExpandBtn');

    // 小于1100px时自动折叠右侧
    if (width < 1100) {
        if (!rightPanel.classList.contains('collapsed')) {
            rightPanel.classList.add('auto-collapse');
            rightExpandBtn.style.display = 'flex';
        }
    } else {
        rightPanel.classList.remove('auto-collapse');
        // 如果不是手动折叠的，隐藏展开按钮
        if (!rightPanel.classList.contains('collapsed')) {
            rightExpandBtn.style.display = 'none';
        }
    }

    // 小于900px时自动折叠左侧
    if (width < 900) {
        if (!leftPanel.classList.contains('collapsed')) {
            leftPanel.classList.add('auto-collapse');
            leftExpandBtn.style.display = 'flex';
        }
    } else {
        leftPanel.classList.remove('auto-collapse');
        // 如果不是手动折叠的，隐藏展开按钮
        if (!leftPanel.classList.contains('collapsed')) {
            leftExpandBtn.style.display = 'none';
        }
    }
}

// 监听窗口大小变化
window.addEventListener('resize', handleResponsiveCollapse);

// 更新缩放显示
function updateZoomDisplay() {
    const zoomBtn = document.querySelector('.canvas-toolbar .small-text');
    if (zoomBtn) {
        zoomBtn.textContent = Math.round(state.zoom * 100) + '%';
    }
}

// 更新光标样式
function updateCursor() {
    if (state.spacePressed) {
        canvas.style.cursor = 'grab';
    } else if (state.currentTool === 'edit') {
        canvas.style.cursor = 'default';
    } else {
        canvas.style.cursor = 'crosshair';
    }
}

function onKeyDown(e) {
    // 如果焦点在输入框中，不处理快捷键（除了ESC）
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
        if (e.key === 'Escape') {
            e.target.blur(); // 退出输入框
        }
        return;
    }

    // ESC键 - 取消当前操作
    if (e.key === 'Escape') {
        e.preventDefault();
        // 优先取消多边形绘制
        if (state.tempPolygon.length > 0) {
            cancelPolygon();
            return;
        }
        // 如果有临时标记，先清除
        if (state.tempBoxes.length > 0 || state.tempPoints.length > 0) {
            clearTempPrompts();
            return;
        }
        // 取消选中
        if (state.selectedAnnotation) {
            state.selectedAnnotation = null;
            updateAnnotationList();
            redraw();
        }
        return;
    }

    // Enter键 - 完成多边形绘制
    if (e.key === 'Enter' && state.tempPolygon.length >= 3) {
        e.preventDefault();
        finishPolygon();
        return;
    }

    // Backspace键 - 撤销多边形最后一个点
    if (e.key === 'Backspace' && state.tempPolygon.length > 0) {
        e.preventDefault();
        undoPolygonPoint();
        return;
    }

    // 空格键 - 进入平移模式
    if (e.code === 'Space' && !state.spacePressed) {
        e.preventDefault();
        state.spacePressed = true;
        updateCursor();
        return;
    }

    if (e.key === 'ArrowLeft') prevImage();
    else if (e.key === 'ArrowRight') nextImage();
    else if (e.key === 's' && e.ctrlKey) {
        e.preventDefault();
        saveAnnotations();
    }
    else if (e.key === 'Delete' && state.selectedAnnotation) {
        deleteSelectedAnnotation();
    }
    // 工具快捷键
    else if (e.key === 'p' || e.key === 'P') setTool('point');
    else if (e.key === 'b' || e.key === 'B') setTool('box');
    else if (e.key === 't' || e.key === 'T') setTool('text');
    else if (e.key === 'e' || e.key === 'E') setTool('edit');
    else if (e.key === 'g' || e.key === 'G') setTool('polygon');
}

function onKeyUp(e) {
    // 空格键释放 - 退出平移模式
    if (e.code === 'Space') {
        state.spacePressed = false;
        // 如果正在拖动，结束拖动
        if (state.panning) {
            state.panning = false;
            state.panStart = null;
        }
        updateCursor();
    }
}

// ==================== 绘制函数 ====================

// 标记静态缓存需要更新
function invalidateStaticCache() {
    staticCacheDirty = true;
}

// 更新静态缓存（图像 + 标注）
function updateStaticCache() {
    if (!currentImage) return;

    const targetWidth = Math.floor(currentImage.width * state.zoom);
    const targetHeight = Math.floor(currentImage.height * state.zoom);

    // 初始化或调整离屏canvas大小
    if (!staticCanvas) {
        staticCanvas = document.createElement('canvas');
        staticCtx = staticCanvas.getContext('2d', { alpha: false });  // 不透明canvas更快
    }

    if (staticCanvas.width !== targetWidth || staticCanvas.height !== targetHeight) {
        staticCanvas.width = targetWidth;
        staticCanvas.height = targetHeight;
    }

    // 优化图像渲染质量设置
    staticCtx.imageSmoothingEnabled = state.zoom < 1;  // 缩小时平滑，放大时锐利
    staticCtx.imageSmoothingQuality = 'medium';

    // 绘制静态内容到离屏canvas
    staticCtx.clearRect(0, 0, staticCanvas.width, staticCanvas.height);
    staticCtx.save();
    staticCtx.scale(state.zoom, state.zoom);
    staticCtx.drawImage(currentImage, 0, 0);

    // 绘制标注
    const originalCtx = ctx;
    ctx = staticCtx;  // 临时切换ctx
    state.annotations.forEach((ann, idx) => {
        drawAnnotation(ann, idx);
    });
    ctx = originalCtx;  // 恢复ctx

    staticCtx.restore();
    staticCacheDirty = false;
}

function redraw() {
    if (!currentImage) return;

    const targetWidth = Math.floor(currentImage.width * state.zoom);
    const targetHeight = Math.floor(currentImage.height * state.zoom);

    // 调整主canvas大小
    if (canvas.width !== targetWidth || canvas.height !== targetHeight) {
        canvas.width = targetWidth;
        canvas.height = targetHeight;
        staticCacheDirty = true;
    }

    // 更新静态缓存（如果需要）
    if (staticCacheDirty) {
        updateStaticCache();
    }

    // 从缓存绘制静态内容
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (staticCanvas) {
        ctx.drawImage(staticCanvas, 0, 0);
    }

    // 绘制动态内容（临时点、框、多边形）
    ctx.save();
    ctx.scale(state.zoom, state.zoom);

    // 绘制临时点
    state.tempPoints.forEach(p => {
        drawPoint(p.x, p.y, p.label);
    });

    // 绘制临时多边形
    if (state.tempPolygon.length > 0) {
        drawTempPolygon();
    }

    ctx.restore();

    // 绘制保存的临时框（在 restore 后绘制）
    state.tempBoxes.forEach(box => {
        drawTempBox(box.x1, box.y1, box.x2, box.y2, box.label);
    });
}

// 快速重绘：只绘制动态内容，用于拖动绘制时
function quickRedraw(dynamicBox) {
    if (!currentImage || !staticCanvas) {
        redraw();
        if (dynamicBox) {
            drawTempBox(dynamicBox.x1, dynamicBox.y1, dynamicBox.x2, dynamicBox.y2);
        }
        return;
    }

    // 从缓存绘制静态内容
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(staticCanvas, 0, 0);

    // 绘制动态内容
    ctx.save();
    ctx.scale(state.zoom, state.zoom);

    state.tempPoints.forEach(p => {
        drawPoint(p.x, p.y, p.label);
    });

    if (state.tempPolygon.length > 0) {
        drawTempPolygon();
    }

    ctx.restore();

    // 绘制已保存的临时框
    state.tempBoxes.forEach(box => {
        drawTempBox(box.x1, box.y1, box.x2, box.y2, box.label);
    });

    // 绘制正在拖动的框
    if (dynamicBox) {
        drawTempBox(dynamicBox.x1, dynamicBox.y1, dynamicBox.x2, dynamicBox.y2);
    }
}

function drawAnnotation(ann, idx) {
    const color = colors[idx % colors.length];
    const isSelected = state.selectedAnnotation === ann.id;
    const number = idx + 1;
    const lineWidth = isSelected ? 3 : 2;

    // 绘制多边形
    if (ann.polygon && ann.polygon.length > 2) {
        ctx.beginPath();
        ctx.moveTo(ann.polygon[0][0], ann.polygon[0][1]);
        for (let i = 1; i < ann.polygon.length; i++) {
            ctx.lineTo(ann.polygon[i][0], ann.polygon[i][1]);
        }
        ctx.closePath();
        ctx.fillStyle = color + '40';
        ctx.fill();
        ctx.strokeStyle = color;
        ctx.lineWidth = lineWidth;
        ctx.stroke();
    }

    // 绘制边界框和简洁标签
    if (ann.bbox) {
        const [x1, y1, x2, y2] = ann.bbox;
        ctx.strokeStyle = color;
        ctx.lineWidth = lineWidth;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

        // 简洁标签: car1, car2
        const label = ann.class_name || ann.label || 'obj';
        const displayText = `${label}${number}`;
        ctx.font = '12px sans-serif';
        const textWidth = ctx.measureText(displayText).width;

        // 小标签背景
        ctx.fillStyle = color;
        ctx.fillRect(x1, y1 - 18, textWidth + 8, 18);

        // 标签文字
        ctx.fillStyle = '#fff';
        ctx.fillText(displayText, x1 + 4, y1 - 5);
    }
}

function drawPoint(x, y, isPositive) {
    ctx.beginPath();
    ctx.arc(x, y, 6, 0, Math.PI * 2);
    ctx.fillStyle = isPositive ? '#4ade80' : '#e94560';
    ctx.fill();
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 2;
    ctx.stroke();
}

function drawTempBox(x1, y1, x2, y2, label) {
    // 应用 zoom 缩放，因为此函数在 ctx.restore() 后调用
    ctx.save();
    ctx.scale(state.zoom, state.zoom);
    // 如果传入了 label 参数（boolean），使用它；否则使用当前的 isPositive 状态
    // 注意：label 可能是 false（负样本），所以要用 typeof 检查
    const isPositive = typeof label === 'boolean' ? label : state.isPositive;
    ctx.strokeStyle = isPositive ? '#4ade80' : '#e94560';
    ctx.lineWidth = 2 / state.zoom;  // 保持线宽视觉一致
    ctx.setLineDash([5 / state.zoom, 5 / state.zoom]);
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    ctx.setLineDash([]);
    ctx.restore();
}

function drawTempPolygon() {
    if (state.tempPolygon.length === 0) return;

    const points = state.tempPolygon;

    // 绘制多边形线条
    ctx.strokeStyle = '#fbbf24';  // 黄色
    ctx.fillStyle = 'rgba(251, 191, 36, 0.2)';
    ctx.lineWidth = 2;
    ctx.setLineDash([]);

    ctx.beginPath();
    ctx.moveTo(points[0][0], points[0][1]);
    for (let i = 1; i < points.length; i++) {
        ctx.lineTo(points[i][0], points[i][1]);
    }

    // 如果超过2个点，闭合并填充
    if (points.length > 2) {
        ctx.closePath();
        ctx.fill();
    }
    ctx.stroke();

    // 绘制顶点
    points.forEach((p, idx) => {
        ctx.beginPath();
        ctx.arc(p[0], p[1], 5, 0, Math.PI * 2);
        ctx.fillStyle = idx === 0 ? '#e94560' : '#fbbf24';  // 第一个点红色
        ctx.fill();
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 2;
        ctx.stroke();
    });

    // 如果有多个点，显示提示连接到第一个点
    if (points.length > 2) {
        ctx.setLineDash([5, 5]);
        ctx.strokeStyle = 'rgba(251, 191, 36, 0.5)';
        ctx.beginPath();
        ctx.moveTo(points[points.length - 1][0], points[points.length - 1][1]);
        ctx.lineTo(points[0][0], points[0][1]);
        ctx.stroke();
        ctx.setLineDash([]);
    }
}

// ==================== 分割操作 ====================

function addPoint(x, y) {
    state.tempPoints.push({ x, y, label: state.isPositive });
    redraw();
    // 不自动分割，等待手动点击分割按钮
    showToast('提示', '点已添加，点击"分割"按钮执行分割');
}

// ==================== 手动多边形绘制 ====================

function addPolygonPoint(x, y) {
    state.tempPolygon.push([x, y]);
    redraw();

    if (state.tempPolygon.length === 1) {
        showToast('提示', '继续点击添加顶点，双击或按Enter完成，ESC取消');
    }
}

function finishPolygon() {
    if (state.tempPolygon.length < 3) {
        showToast('提示', '多边形至少需要3个顶点');
        return;
    }

    // 检查类别
    const className = getCurrentClassName();
    if (!className) return;

    // 计算边界框
    const xs = state.tempPolygon.map(p => p[0]);
    const ys = state.tempPolygon.map(p => p[1]);
    const x1 = Math.min(...xs);
    const y1 = Math.min(...ys);
    const x2 = Math.max(...xs);
    const y2 = Math.max(...ys);

    // 计算面积 (Shoelace formula)
    let area = 0;
    for (let i = 0; i < state.tempPolygon.length; i++) {
        const j = (i + 1) % state.tempPolygon.length;
        area += state.tempPolygon[i][0] * state.tempPolygon[j][1];
        area -= state.tempPolygon[j][0] * state.tempPolygon[i][1];
    }
    area = Math.abs(area) / 2;

    // 创建标注
    const annotation = {
        id: generateId(),
        label: className,
        class_name: className,
        score: 1.0,  // 手动标注置信度为1
        bbox: [x1, y1, x2, y2],
        polygon: state.tempPolygon.slice(),  // 复制数组
        area: area,
        manual: true  // 标记为手动绘制
    };

    state.annotations.push(annotation);
    state.tempPolygon = [];
    invalidateStaticCache();  // 标注变化，更新缓存
    updateAnnotationList();
    redraw();
    showToast('成功', `已添加手动标注: ${className}`);
}

function cancelPolygon() {
    if (state.tempPolygon.length > 0) {
        state.tempPolygon = [];
        redraw();
        showToast('提示', '已取消多边形绘制');
    }
}

function undoPolygonPoint() {
    if (state.tempPolygon.length > 0) {
        state.tempPolygon.pop();
        redraw();
    }
}

function generateId() {
    return Math.random().toString(36).substr(2, 8);
}

// 手动触发分割（点击分割按钮）
async function segmentManual() {
    if (!state.projectId || state.currentIndex < 0) {
        showToast('提示', '请先选择项目和图片');
        return;
    }

    // 优先分割临时框
    if (state.tempBoxes.length > 0) {
        await segmentByBoxes(state.tempBoxes);
        state.tempBoxes = [];
        return;
    }

    // 其次分割临时点
    if (state.tempPoints.length > 0) {
        await segmentByPoints();
        return;
    }

    showToast('提示', '请先绘制框或添加点');
}

// 清除临时提示（框和点）
function clearTempPrompts() {
    state.tempBoxes = [];
    state.tempPoints = [];
    redraw();
    showToast('提示', '已清除临时标记');
}

async function segmentByPoints() {
    if (!state.projectId || state.currentIndex < 0) return;

    // 检查类别
    const className = getCurrentClassName();
    if (!className) return;

    const image = state.images[state.currentIndex];
    const points = state.tempPoints.map(p => [p.x, p.y, p.label ? 1 : 0]);

    showLoading('正在分割...');

    try {
        const response = await fetch('/api/segment/point', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                image_id: image.id,
                points: points
            })
        });

        const data = await response.json();
        if (data.success && data.results.length > 0) {
            data.results.forEach(r => r.class_name = className);

            // 增量保存到数据库（不删除已有标注）
            await fetch('/api/annotation/add', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    project_id: state.projectId,
                    image_index: state.currentIndex,
                    annotations: data.results,
                    class_name: className
                })
            });

            // 更新前端状态
            state.annotations.push(...data.results);
            state.tempPoints = [];
            invalidateStaticCache();  // 标注变化，更新缓存
            updateAnnotationList();
            redraw();

            // 更新图片标注状态
            if (state.images[state.currentIndex]) {
                state.images[state.currentIndex].annotated = true;
            }
            updateImageList();

            showToast('成功', `检测到 ${data.results.length} 个 "${className}"`);
        } else {
            showToast('提示', '未检测到对象');
        }
    } catch (error) {
        showToast('错误', error.message, 'danger');
    }

    hideLoading();
}

async function segmentByBoxes(boxes) {
    if (!state.projectId || state.currentIndex < 0) return;

    // 检查类别
    const className = getCurrentClassName();
    if (!className) return;

    const image = state.images[state.currentIndex];

    // 转换框数据，每个框使用自己存储的 label
    const boxData = boxes.map(box => [box.x1, box.y1, box.x2, box.y2, box.label ? 1 : 0]);

    // 统计正负样本数量
    const positiveCount = boxes.filter(b => b.label).length;
    const negativeCount = boxes.length - positiveCount;
    console.log(`[DEBUG] 分割框: 正样本=${positiveCount}, 负样本=${negativeCount}`);

    showLoading('正在分割...');

    try {
        const response = await fetch('/api/segment/box', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                image_id: image.id,
                boxes: boxData
            })
        });

        const data = await response.json();
        if (data.success && data.results.length > 0) {
            data.results.forEach(r => r.class_name = className);

            // 增量保存到数据库（不删除已有标注）
            await fetch('/api/annotation/add', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    project_id: state.projectId,
                    image_index: state.currentIndex,
                    annotations: data.results,
                    class_name: className
                })
            });

            // 更新前端状态
            state.annotations.push(...data.results);
            invalidateStaticCache();  // 标注变化，更新缓存
            updateAnnotationList();
            redraw();

            // 更新图片标注状态
            if (state.images[state.currentIndex]) {
                state.images[state.currentIndex].annotated = true;
            }
            updateImageList();

            showToast('成功', `检测到 ${data.results.length} 个 "${className}"`);
        } else {
            showToast('提示', '未检测到对象');
        }
    } catch (error) {
        showToast('错误', error.message, 'danger');
    }

    hideLoading();
}

// 获取当前选中的类名
function getCurrentClassName() {
    if (state.classes.length === 0) {
        showToast('提示', '请先在类别管理中添加类名');
        return null;
    }

    // 如果有当前选中的类名，使用它
    if (state.currentClass && state.classes.includes(state.currentClass)) {
        return state.currentClass;
    }

    // 如果只有一个类名，直接使用
    if (state.classes.length === 1) {
        return state.classes[0];
    }

    // 多个类名，提示用户选择
    showToast('提示', '请先在类别管理中点击选择一个类名');
    return null;
}

async function segmentByText() {
    if (!state.projectId || state.currentIndex < 0) {
        console.log('[DEBUG] segmentByText: no project or image selected');
        showToast('提示', '请先选择项目和图片');
        return;
    }

    // 检查类别列表
    if (state.classes.length === 0) {
        showToast('提示', '请先在类别管理中添加类名');
        return;
    }

    const prompt = document.getElementById('textPrompt').value.trim();
    if (!prompt) {
        showToast('提示', '请输入分割提示词');
        return;
    }

    // 匹配类名：优先精确匹配，其次模糊匹配，否则使用当前选中的类名
    let matchedClass = findMatchingClass(prompt);

    if (!matchedClass) {
        // 没有匹配到，提示用户选择
        if (state.classes.length === 1) {
            matchedClass = state.classes[0];
        } else {
            showClassSelectModal(prompt);
            return;
        }
    }

    await executeTextSegment(prompt, matchedClass);
}

// 查找匹配的类名
function findMatchingClass(prompt) {
    const promptLower = prompt.toLowerCase();

    // 精确匹配
    for (const cls of state.classes) {
        if (cls.toLowerCase() === promptLower) {
            return cls;
        }
    }

    // 包含匹配（提示词包含类名，或类名包含提示词）
    for (const cls of state.classes) {
        const clsLower = cls.toLowerCase();
        if (promptLower.includes(clsLower) || clsLower.includes(promptLower)) {
            return cls;
        }
    }

    return null;
}

// 显示类名选择弹窗
function showClassSelectModal(prompt) {
    // 创建选择弹窗
    const modalHtml = `
        <div class="modal fade" id="classSelectModal" tabindex="-1">
            <div class="modal-dialog modal-sm">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title">选择类别</h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body">
                        <p class="small text-muted mb-3">提示词 "${prompt}" 未匹配到类别，请选择：</p>
                        <div class="class-select-list">
                            ${state.classes.map((cls, idx) => `
                                <button class="btn btn-outline-primary w-100 mb-2 class-select-btn"
                                        data-class="${cls}" data-prompt="${prompt}">
                                    ${cls}
                                </button>
                            `).join('')}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    `;

    // 移除旧的弹窗
    const oldModal = document.getElementById('classSelectModal');
    if (oldModal) oldModal.remove();

    // 添加新弹窗
    document.body.insertAdjacentHTML('beforeend', modalHtml);

    // 绑定点击事件
    document.querySelectorAll('.class-select-btn').forEach(btn => {
        btn.addEventListener('click', async () => {
            const selectedClass = btn.dataset.class;
            const prompt = btn.dataset.prompt;
            bootstrap.Modal.getInstance(document.getElementById('classSelectModal')).hide();
            await executeTextSegment(prompt, selectedClass);
        });
    });

    // 显示弹窗
    new bootstrap.Modal(document.getElementById('classSelectModal')).show();
}

// 执行文本分割
async function executeTextSegment(prompt, className) {
    const image = state.images[state.currentIndex];

    // 尝试AI翻译（如果启用）
    let actualPrompt = prompt;
    let wasTranslated = false;

    if (aiConfig.enabled && aiConfig.apiUrl && aiConfig.apiKey) {
        showLoading(`正在翻译: ${prompt}...`);
        const translateResult = await translatePrompt(prompt);
        if (translateResult.translated) {
            actualPrompt = translateResult.text;
            wasTranslated = true;
            console.log(`[AI翻译] "${prompt}" -> "${actualPrompt}"`);
        }
    }

    console.log('[DEBUG] segmentByText:', { image_id: image.id, prompt: actualPrompt, className, confidence: state.confidence });

    showLoading(`正在分割: ${wasTranslated ? actualPrompt + ' (' + prompt + ')' : prompt}...`);

    try {
        const response = await fetch('/api/segment/text', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                image_id: image.id,
                prompt: actualPrompt,
                confidence: state.confidence
            })
        });

        const data = await response.json();
        console.log('[DEBUG] segmentByText response:', data);

        if (data.success) {
            if (data.results && data.results.length > 0) {
                console.log('[DEBUG] Found results:', data.results);
                // 使用匹配的类名，而不是提示词
                data.results.forEach(r => r.class_name = className);

                // 增量保存到数据库（不删除已有标注）
                await fetch('/api/annotation/add', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        project_id: state.projectId,
                        image_index: state.currentIndex,
                        annotations: data.results,
                        class_name: className
                    })
                });

                // 更新前端状态
                state.annotations.push(...data.results);
                invalidateStaticCache();  // 标注变化，更新缓存
                updateAnnotationList();
                redraw();

                // 更新图片标注状态
                if (state.images[state.currentIndex]) {
                    state.images[state.currentIndex].annotated = true;
                }
                updateImageList();

                const msg = wasTranslated
                    ? `检测到 ${data.results.length} 个 "${className}" (翻译: ${actualPrompt})`
                    : `检测到 ${data.results.length} 个 "${className}"`;
                showToast('成功', msg);
            } else {
                const msg = wasTranslated
                    ? `未检测到 "${prompt}" (翻译: ${actualPrompt})`
                    : `未检测到 "${prompt}"`;
                showToast('提示', msg);
            }
        }
    } catch (error) {
        showToast('错误', error.message, 'danger');
    }

    hideLoading();
}

async function batchSegment() {
    console.log('[DEBUG batchSegment] 开始批量分割');
    console.log('[DEBUG] projectId:', state.projectId);
    console.log('[DEBUG] images.length:', state.images.length);
    console.log('[DEBUG] classes:', state.classes);
    console.log('[DEBUG] images data sample:', state.images.slice(0, 3));

    if (!state.projectId) {
        showToast('提示', '请先选择项目');
        return;
    }

    // 检查类别
    if (state.classes.length === 0) {
        showToast('提示', '请先在类别管理中添加类名');
        return;
    }

    const prompt = document.getElementById('textPrompt').value.trim();
    if (!prompt) {
        showToast('提示', '请输入提示词');
        return;
    }

    // 匹配类名
    let className = findMatchingClass(prompt);
    if (!className) {
        if (state.classes.length === 1) {
            className = state.classes[0];
        } else if (state.currentClass) {
            className = state.currentClass;
        } else {
            showToast('提示', `提示词 "${prompt}" 未匹配到类别，请先选择一个类别`);
            return;
        }
    }

    const startIndex = parseInt(document.getElementById('batchStart').value) || 0;
    const endIndex = parseInt(document.getElementById('batchEnd').value) || -1;
    const skipAnnotated = document.getElementById('skipAnnotated').checked;
    const batchMode = document.getElementById('batchMode').value;  // 'serial' 或 'concurrent'

    // 计算要处理的图片数量
    const totalImages = state.images.length;
    const actualEnd = endIndex === -1 ? totalImages : Math.min(endIndex, totalImages);

    // 构建待处理列表
    const toProcessList = [];
    for (let i = startIndex; i < actualEnd; i++) {
        const img = state.images[i];
        const isAnnotated = img && img.annotated;
        if (skipAnnotated && isAnnotated) continue;
        toProcessList.push(i);
    }

    if (toProcessList.length === 0) {
        showToast('提示', '没有需要处理的图片');
        return;
    }

    // 尝试AI翻译（如果启用）
    let actualPrompt = prompt;
    let wasTranslated = false;

    if (aiConfig.enabled && aiConfig.apiUrl && aiConfig.apiKey) {
        showLoading(`正在翻译: ${prompt}...`);
        const translateResult = await translatePrompt(prompt);
        if (translateResult.translated) {
            actualPrompt = translateResult.text;
            wasTranslated = true;
            console.log(`[AI翻译] "${prompt}" -> "${actualPrompt}"`);
        }
    }

    // 获取最大并发数（仅并发模式）
    const maxWorkers = batchMode === 'concurrent'
        ? (parseInt(document.getElementById('maxWorkers').value) || 2)
        : 2;

    // 确认操作（根据模式显示不同提示）
    const modeText = batchMode === 'concurrent'
        ? `并发批量处理 (最大${maxWorkers}线程 + 特征缓存)`
        : '单线程串行处理 (慢)';
    
    // 如果图片数量超过 100 张，添加警告提示
    let warningText = '';
    if (toProcessList.length > 100) {
        warningText = '\n\n⚠️ 警告: 图片数量超过 100 张，可能导致请求超时。\n建议分批处理，每次不超过 100 张。';
    }
    
    const confirmMsg = wasTranslated
        ? `即将对 ${toProcessList.length} 张图片进行批量分割\n模式: ${modeText}\n类别: ${className}\n提示词: ${prompt}\n翻译后: ${actualPrompt}${warningText}\n\n确定继续？`
        : `即将对 ${toProcessList.length} 张图片进行批量分割\n模式: ${modeText}\n类别: ${className}\n提示词: ${prompt}${warningText}\n\n确定继续？`;

    if (!confirm(confirmMsg)) {
        return;
    }

    // 根据模式选择处理方式
    if (batchMode === 'concurrent') {
        await batchSegmentConcurrent(toProcessList, className, actualPrompt, actualPrompt !== prompt);
    } else {
        await batchSegmentSerial(toProcessList, className, actualPrompt, actualPrompt !== prompt);
    }

    // 刷新显示
    updateImageList();
    if (state.currentIndex >= 0) {
        loadImage(state.currentIndex);
    }
}

// 单线程串行批量处理
async function batchSegmentSerial(toProcessList, className, actualPrompt, wasTranslated) {
    let processed = 0;
    let failed = 0;
    let totalDetections = 0;

    showLoading(`正在批量分割 (串行模式)... 0/${toProcessList.length}`, 0);

    for (let i = 0; i < toProcessList.length; i++) {
        const imgIndex = toProcessList[i];
        const img = state.images[imgIndex];

        // 更新进度
        const progress = (i / toProcessList.length) * 100;
        showLoading(`正在处理: ${img.filename} (${i + 1}/${toProcessList.length}) [串行]`, progress);

        try {
            const response = await fetch('/api/segment/text', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    image_id: img.id,
                    prompt: actualPrompt,
                    confidence: state.confidence
                })
            });

            const data = await response.json();
            if (data.success && data.results && data.results.length > 0) {
                // 设置类名
                data.results.forEach(r => r.class_name = className);

                // 增量保存标注（不删除已有标注）
                await fetch('/api/annotation/add', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        project_id: state.projectId,
                        image_index: imgIndex,
                        annotations: data.results,
                        class_name: className
                    })
                });

                // 更新本地状态（追加标注）
                const existingAnnotations = state.images[imgIndex].annotations || [];
                state.images[imgIndex].annotations = [...existingAnnotations, ...data.results];
                state.images[imgIndex].annotated = true;
                totalDetections += data.results.length;
            }
            processed++;
        } catch (error) {
            console.error(`处理图片 ${img.filename} 失败:`, error);
            failed++;
        }
    }

    // 完成
    updateLoadingProgress(100);
    hideLoading();

    // 刷新图片列表显示
    updateImageList();

    const msg = `处理完成!\n成功: ${processed} 张\n检测到: ${totalDetections} 个对象`;
    if (failed > 0) {
        showToast('警告', msg + `\n失败: ${failed} 张`, 'warning');
    } else {
        showToast('成功', msg);
    }
}

// 并发批量处理（异步优化版）
async function batchSegmentConcurrent(toProcessList, className, actualPrompt, wasTranslated) {
    const maxWorkers = parseInt(document.getElementById('maxWorkers').value) || 2;
    const taskId = Date.now().toString() + Math.random().toString(36).substr(2, 9);

    showLoading(`正在启动批量分割 (并发模式, 最大${maxWorkers}线程)...`, 0);

    // 启动进度轮询
    let progressInterval = null;

    const pollProgress = async () => {
        try {
            const progressResponse = await fetch(`/api/segment/batch/progress?task_id=${taskId}`);
            const progressData = await progressResponse.json();

            if (progressData.success && progressData.progress) {
                const { current, total, status } = progressData.progress;
                const percentage = total > 0 ? Math.round((current / total) * 100) : 0;

                if (status === 'processing') {
                    updateLoadingProgress(percentage);
                    document.getElementById('loadingText').textContent =
                        `正在批量分割 (并发模式, 最大${maxWorkers}线程)... ${current}/${total} (${percentage}%)`;
                } else if (status === 'done') {
                    // 处理完成，获取最终结果
                    clearInterval(progressInterval);
                    updateLoadingProgress(100);

                    // 显示最终统计信息
                    const progress = progressData.progress;
                    const msg = `处理完成!\n成功: ${progress.processed} 张\n检测到: ${progress.total_detections} 个对象\n总耗时: ${progress.total_time}秒\n平均: ${progress.avg_time_per_image}秒/张`;

                    hideLoading();

                    // 刷新图片列表显示
                    updateImageList();
                    if (state.currentIndex >= 0) {
                        loadImage(state.currentIndex);
                    }

                    showToast('成功', msg);
                } else if (status === 'error') {
                    clearInterval(progressInterval);
                    hideLoading();
                    showToast('错误', progressData.progress.message || '处理失败', 'error');
                    return;
                }
            }
        } catch (error) {
            console.error('获取进度失败:', error);
        }
    };

    try {
        // 发起批量处理请求（立即返回task_id）
        console.log(`[批量处理] 待处理图片数量: ${toProcessList.length}`);
        console.log(`[批量处理] 图片索引: ${toProcessList.slice(0, 10)}${toProcessList.length > 10 ? '...' : ''}`);
        console.log(`[批量处理] start_index=${toProcessList[0]}, end_index=${toProcessList[toProcessList.length - 1] + 1}`);

        const response = await fetch('/api/segment/batch/concurrent', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                project_id: state.projectId,
                prompt: actualPrompt,
                class_name: className,
                start_index: toProcessList[0],
                end_index: toProcessList[toProcessList.length - 1] + 1,
                skip_annotated: document.getElementById('skipAnnotated').checked,
                confidence: state.confidence,
                use_cache: true,
                max_workers: maxWorkers,
                task_id: taskId
            })
        });

        // 检查响应状态
        if (!response.ok) {
            throw new Error(`服务器返回错误: ${response.status} ${response.statusText}`);
        }

        const contentType = response.headers.get('content-type');
        if (!contentType || !contentType.includes('application/json')) {
            const text = await response.text();
            throw new Error(`服务器返回了非 JSON 响应: ${text.substring(0, 200)}...`);
        }

        const data = await response.json();

        if (!data.success) {
            hideLoading();
            showToast('错误', `批量处理失败: ${data.error}`, 'danger');
            return;
        }

        // 任务已成功启动，开始轮询进度
        updateLoadingProgress(-1);  // 显示不确定进度动画
        document.getElementById('loadingText').textContent =
            `批量处理已启动，正在初始化... (任务ID: ${taskId})`;

        // 立即查询一次，然后每1500ms查询一次
        setTimeout(pollProgress, 500);
        progressInterval = setInterval(pollProgress, 1500);

    } catch (error) {
        console.error('并发批量处理失败:', error);
        if (progressInterval) {
            clearInterval(progressInterval);
        }
        hideLoading();

        if (error.message.includes('服务器返回错误') || error.message.includes('非 JSON 响应')) {
            showToast('错误', `批量处理失败: 服务器响应异常，请检查服务器日志`, 'danger');
        } else {
            showToast('错误', `批量处理失败: ${error.message}`, 'danger');
        }
    }
}

// ==================== 项目管理 ====================

function showProjectModal() {
    loadProjects();
    new bootstrap.Modal(document.getElementById('projectModal')).show();
}

async function loadProjects() {
    try {
        const response = await fetch('/api/project/list');
        const data = await response.json();

        const list = document.getElementById('projectList');
        if (data.projects.length === 0) {
            list.innerHTML = '<div class="text-muted p-3">暂无项目</div>';
            return;
        }

        list.innerHTML = data.projects.map(p => `
            <div class="list-group-item project-item">
                <div class="project-item-main" onclick="selectProject('${p.id}')">
                    <div class="d-flex justify-content-between align-items-center">
                        <strong>${p.name}</strong>
                        <small class="text-muted">${p.image_count || 0} 张图片</small>
                    </div>
                    <small class="text-muted">${p.image_dir || '未设置目录'}</small>
                </div>
                <div class="project-item-actions">
                    <button class="btn btn-sm btn-outline-primary" onclick="event.stopPropagation(); showEditProjectModal('${p.id}')" title="编辑">
                        <i class="bi bi-pencil"></i>
                    </button>
                    <button class="btn btn-sm btn-outline-danger" onclick="event.stopPropagation(); deleteProject('${p.id}', '${p.name}')" title="删除">
                        <i class="bi bi-trash"></i>
                    </button>
                </div>
            </div>
        `).join('');
    } catch (error) {
        console.error('加载项目失败:', error);
    }
}

// 编辑项目状态
let editingProjectId = null;

// 显示编辑项目模态框
async function showEditProjectModal(projectId) {
    try {
        const response = await fetch(`/api/project/${projectId}`);
        const data = await response.json();

        if (!data.success) {
            showToast('错误', data.error, 'danger');
            return;
        }

        const project = data.project;
        editingProjectId = projectId;

        // 切换到新建项目标签页（复用表单）
        const tab = document.querySelector('a[href="#newProject"]');
        if (tab) {
            new bootstrap.Tab(tab).show();
        }

        // 填充表单
        document.getElementById('newProjectName').value = project.name || '';
        document.getElementById('newProjectImageDir').value = project.image_dir || '';
        document.getElementById('newProjectOutputDir').value = project.output_dir || '';
        document.getElementById('newProjectClasses').value = (project.classes || []).join(', ');

        // 修改按钮文字
        const createBtn = document.querySelector('#newProject button.btn-primary');
        if (createBtn) {
            createBtn.innerHTML = '<i class="bi bi-check-lg"></i> 保存修改';
            createBtn.onclick = updateProject;
        }

        // 修改标签页标题
        const tabLink = document.querySelector('a[href="#newProject"]');
        if (tabLink) {
            tabLink.textContent = '编辑项目';
        }

    } catch (error) {
        showToast('错误', error.message, 'danger');
    }
}

// 更新项目
async function updateProject() {
    if (!editingProjectId) {
        createProject();
        return;
    }

    const name = document.getElementById('newProjectName').value.trim();
    const imageDir = document.getElementById('newProjectImageDir').value.trim();
    const outputDir = document.getElementById('newProjectOutputDir').value.trim();
    const classesStr = document.getElementById('newProjectClasses').value.trim();

    if (!name) {
        showToast('提示', '请输入项目名称');
        return;
    }

    const classes = classesStr ? classesStr.split(',').map(c => c.trim()).filter(c => c) : [];

    try {
        const response = await fetch(`/api/project/${editingProjectId}/update`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                name,
                image_dir: imageDir,
                output_dir: outputDir,
                classes
            })
        });

        const data = await response.json();
        if (data.success) {
            showToast('成功', '项目已更新');

            // 如果是当前项目，更新显示
            if (state.projectId === editingProjectId) {
                document.getElementById('projectName').textContent = name;
                state.classes = classes;
                updateClassList();
            }

            // 重置编辑状态
            resetProjectForm();
            loadProjects();
        } else {
            showToast('错误', data.error, 'danger');
        }
    } catch (error) {
        showToast('错误', error.message, 'danger');
    }
}

// 删除项目
async function deleteProject(projectId, projectName) {
    if (!confirm(`确定要删除项目 "${projectName}" 吗？\n\n此操作将删除项目及其所有标注数据，不可恢复！`)) {
        return;
    }

    try {
        const response = await fetch(`/api/project/${projectId}/delete`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });

        const data = await response.json();
        if (data.success) {
            showToast('成功', '项目已删除');

            // 如果删除的是当前项目，清空状态
            if (state.projectId === projectId) {
                state.projectId = null;
                state.images = [];
                state.annotations = [];
                state.classes = [];
                state.currentIndex = 0;
                document.getElementById('projectName').textContent = '未选择项目';
                updateImageList();
                updateClassList();
                updateAnnotationList();
                localStorage.removeItem('sam3_work_state');

                // 清空画布
                if (currentImage) {
                    currentImage = null;
                    const canvas = document.getElementById('annotationCanvas');
                    const ctx = canvas.getContext('2d');
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                }
            }

            loadProjects();
        } else {
            showToast('错误', data.error, 'danger');
        }
    } catch (error) {
        showToast('错误', error.message, 'danger');
    }
}

// 重置项目表单
function resetProjectForm() {
    editingProjectId = null;
    document.getElementById('newProjectName').value = '';
    document.getElementById('newProjectImageDir').value = '';
    document.getElementById('newProjectOutputDir').value = '';
    document.getElementById('newProjectClasses').value = '';

    // 恢复按钮文字
    const createBtn = document.querySelector('#newProject button.btn-primary');
    if (createBtn) {
        createBtn.innerHTML = '<i class="bi bi-plus-circle"></i> 创建项目';
        createBtn.onclick = createProject;
    }

    // 恢复标签页标题
    const tabLink = document.querySelector('a[href="#newProject"]');
    if (tabLink) {
        tabLink.textContent = '新建项目';
    }
}

// 监听模态框关闭事件，重置表单
document.addEventListener('DOMContentLoaded', () => {
    const projectModal = document.getElementById('projectModal');
    if (projectModal) {
        projectModal.addEventListener('hidden.bs.modal', resetProjectForm);
    }
});

async function createProject() {
    // 如果是编辑模式，调用更新函数
    if (editingProjectId) {
        updateProject();
        return;
    }

    const name = document.getElementById('newProjectName').value.trim();
    const imageDir = document.getElementById('newProjectImageDir').value.trim();
    const outputDir = document.getElementById('newProjectOutputDir').value.trim();
    const classesStr = document.getElementById('newProjectClasses').value.trim();

    if (!name) {
        showToast('提示', '请输入项目名称');
        return;
    }

    const classes = classesStr ? classesStr.split(',').map(c => c.trim()).filter(c => c) : [];

    try {
        const response = await fetch('/api/project/create', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name, image_dir: imageDir, output_dir: outputDir, classes })
        });

        const data = await response.json();
        if (data.success) {
            showToast('成功', '项目创建成功');
            await selectProject(data.project.id);

            if (imageDir) {
                await loadProjectImages();
            }

            bootstrap.Modal.getInstance(document.getElementById('projectModal')).hide();
        }
    } catch (error) {
        showToast('错误', error.message, 'danger');
    }
}

async function selectProject(projectId) {
    try {
        // 使用轻量级接口获取项目信息和图片列表（不含标注数据）
        const [infoResponse, imagesResponse] = await Promise.all([
            fetch(`/api/project/${projectId}/info`),
            fetch(`/api/project/${projectId}/images`)
        ]);

        const infoData = await infoResponse.json();
        const imagesData = await imagesResponse.json();

        if (infoData.success && imagesData.success) {
            state.projectId = projectId;
            state.classes = infoData.project.classes || [];
            state.images = imagesData.images || [];
            state.currentIndex = 0;

            document.getElementById('projectName').textContent = infoData.project.name;
            document.getElementById('exportOutputDir').value = infoData.project.output_dir || '';

            updateClassList();
            updateImageList();

            // 保存工作状态
            saveWorkState();

            bootstrap.Modal.getInstance(document.getElementById('projectModal'))?.hide();

            // 返回 true 表示成功，供 restoreWorkState 使用
            return true;
        }
    } catch (error) {
        showToast('错误', error.message, 'danger');
    }
    return false;
}

async function loadProjectImages() {
    if (!state.projectId) return;

    const imageDir = document.getElementById('newProjectImageDir').value.trim();
    if (!imageDir) return;

    try {
        const response = await fetch(`/api/project/${state.projectId}/load_images`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image_dir: imageDir })
        });

        if (!response.ok) {
            console.error('加载图片失败:', response.status, response.statusText);
            const errorText = await response.text();
            console.error('错误详情:', errorText);
            showToast('错误', '加载图片失败', 'danger');
            return;
        }

        const data = await response.json();
        if (data.success) {
            state.images = data.images;
            updateImageList();
            showToast('成功', `已加载 ${data.count} 张图片`);
        }
    } catch (error) {
        showToast('错误', error.message, 'danger');
    }
}

// ==================== 图片导航 ====================

function updateImageList() {
    const list = document.getElementById('imageList');
    const annotatedCount = state.images.filter(img => img.annotated).length;
    const total = state.images.length;
    const percent = total > 0 ? Math.round(annotatedCount / total * 100) : 0;

    // 更新统计
    document.getElementById('imageStats').textContent = `${annotatedCount}/${total}`;

    // 更新进度条
    document.getElementById('progressBar').style.width = `${percent}%`;
    document.getElementById('progressText').textContent = `已标注: ${annotatedCount}/${total} (${percent}%)`;

    if (total === 0) {
        list.innerHTML = '<div class="empty-state"><i class="bi bi-images"></i><p>暂无图片</p></div>';
        return;
    }

    list.innerHTML = state.images.map((img, idx) => `
        <div class="image-item ${idx === state.currentIndex ? 'active' : ''} ${img.annotated ? 'annotated' : ''}"
             onclick="loadImage(${idx})">
            <span class="index">${idx + 1}</span>
            <span class="filename">${img.filename}</span>
        </div>
    `).join('');
}

function filterImages(query) {
    const items = document.querySelectorAll('.image-item');
    query = query.toLowerCase();
    items.forEach((item, idx) => {
        const filename = state.images[idx].filename.toLowerCase();
        item.style.display = filename.includes(query) ? '' : 'none';
    });
}

// 筛选低置信度图片
async function filterLowConfidenceImages() {
    if (!state.projectId) {
        showToast('错误', '请先打开项目');
        return;
    }

    showToast('提示', '正在筛选低置信度标注图片...');

    try {
        // 获取所有类别的统计
        const response = await fetch('/api/annotation/stats_by_class', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                project_id: state.projectId
            })
        });

        const result = await response.json();
        if (!result.success) {
            showToast('错误', result.error);
            return;
        }

        // 获取每张图片的低置信度标注数量
        const threshold = state.filterConfidence;
        const lowConfidenceImages = [];

        // 遍历所有图片，统计低置信度标注数量
        for (let i = 0; i < state.images.length; i++) {
            const img = state.images[i];
            try {
                const annResponse = await fetch('/api/image/annotations', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        project_id: state.projectId,
                        image_index: i
                    })
                });

                const annResult = await annResponse.json();
                if (annResult.success && annResult.annotations) {
                    const lowConfCount = annResult.annotations.filter(a =>
                        (!a.manual || a.manual === 0) && a.score !== null && a.score < threshold
                    ).length;

                    if (lowConfCount > 0) {
                        lowConfidenceImages.push({
                            index: i,
                            filename: img.filename,
                            lowConfCount: lowConfCount
                        });
                    }
                }
            } catch (error) {
                console.error(`获取图片 ${i} 标注失败:`, error);
            }
        }

        if (lowConfidenceImages.length === 0) {
            showToast('提示', `没有找到置信度低于 ${threshold.toFixed(2)} 的标注图片`);
            return;
        }

        // 更新图片列表，只显示含有低置信度标注的图片
        state.filteredImages = lowConfidenceImages.map(item => item.index);
        updateImageListWithFilter(state.filteredImages);

        showToast('成功', `找到 ${lowConfidenceImages.length} 张含有低置信度标注的图片`);

    } catch (error) {
        console.error('筛选低置信度图片失败:', error);
        showToast('错误', '筛选失败');
    }
}

// 更新图片列表（带筛选）
function updateImageListWithFilter(filteredIndices) {
    const list = document.getElementById('imageList');

    // 更新进度条（只统计筛选后的）
    const total = filteredIndices.length;
    const annotatedCount = filteredIndices.filter(idx => state.images[idx]?.annotated).length;
    const percent = total > 0 ? Math.round(annotatedCount / total * 100) : 0;

    // 更新进度条
    document.getElementById('progressBar').style.width = `${percent}%`;
    document.getElementById('progressText').textContent = `已标注: ${annotatedCount}/${total} (${percent}%)`;
    document.getElementById('imageStats').textContent = `${total}/${state.images.length}`;

    list.innerHTML = filteredIndices.map(idx => {
        const img = state.images[idx];
        return `
            <div class="image-item ${idx === state.currentIndex ? 'active' : ''} ${img.annotated ? 'annotated' : ''} has-low-confidence"
                 onclick="loadImage(${idx})">
                <span class="index">${idx + 1}</span>
                <span class="filename">${img.filename}</span>
                <span class="badge bg-warning position-absolute top-0 end-0 m-1" style="font-size: 10px;">低置信</span>
            </div>
        `;
    }).join('');
}

// 清除图片筛选
function clearImageFilter() {
    state.filteredImages = [];
    updateImageList();
    showToast('提示', '已清除筛选，显示所有图片');
}

async function loadImage(index) {
    if (index < 0 || index >= state.images.length) return;

    // 保存当前标注（不标记为手动）
    if (state.currentIndex !== index && state.annotations.length > 0) {
        await saveAnnotations(false, false);
    }

    state.currentIndex = index;
    state.annotations = [];
    state.tempPoints = [];
    state.tempBoxes = [];
    state.selectedAnnotation = null;
    invalidateStaticCache();  // 加载新图片，重置缓存

    const image = state.images[index];

    // 从API加载已有标注（使用图片ID而不是索引）
    if (state.projectId && image.id) {
        try {
            const response = await fetch(`/api/annotation/get?project_id=${state.projectId}&image_id=${image.id}`);
            const data = await response.json();
            if (data.success) {
                state.annotations = data.annotations || [];
                // 更新 annotated 状态以反映实际的标注数据
                image.annotated = state.annotations.length > 0;
            }
        } catch (e) {
            console.error('加载标注失败:', e);
        }
    }

    // 保存工作状态
    saveWorkState();

    // 加载图片
    currentImage = new Image();
    currentImage.onload = () => {
        // 首次加载时适应视图
        fitToView();
        updateAnnotationList();
        updateImageList();
        redraw();  // 重绘画布以显示标注
        document.getElementById('currentImageInfo').textContent =
            `${index + 1} / ${state.images.length} - ${image.filename}`;
    };
    currentImage.src = `/api/image/serve?image_id=${image.id}`;
}

function prevImage() {
    if (state.currentIndex > 0) {
        loadImage(state.currentIndex - 1);
    }
}

function nextImage() {
    if (state.currentIndex < state.images.length - 1) {
        loadImage(state.currentIndex + 1);
    }
}

// ==================== 标注管理 ====================

function updateAnnotationList() {
    const list = document.getElementById('annotationList');

    if (state.annotations.length === 0) {
        list.innerHTML = '<div class="text-muted small p-2">暂无标注</div>';
        return;
    }

    list.innerHTML = state.annotations.map((ann, idx) => `
        <div class="annotation-item ${state.selectedAnnotation === ann.id ? 'selected' : ''}"
             onclick="selectAnnotation('${ann.id}')">
            <div class="color-indicator" style="background-color: ${colors[idx % colors.length]}"></div>
            <div class="info">
                <div class="label">${ann.class_name || ann.label || 'obj'}${idx + 1}</div>
                <div class="score">置信度: ${(ann.score || 0).toFixed(2)}</div>
            </div>
            <div class="actions">
                <button class="btn btn-outline-danger btn-sm" onclick="deleteAnnotation('${ann.id}', event)">
                    <i class="bi bi-trash"></i>
                </button>
            </div>
        </div>
    `).join('');
}

function selectAnnotation(id) {
    state.selectedAnnotation = state.selectedAnnotation === id ? null : id;
    updateAnnotationList();
    redraw();
}

function selectAnnotationAt(x, y) {
    for (let i = state.annotations.length - 1; i >= 0; i--) {
        const ann = state.annotations[i];
        if (ann.bbox) {
            const [x1, y1, x2, y2] = ann.bbox;
            if (x >= x1 && x <= x2 && y >= y1 && y <= y2) {
                selectAnnotation(ann.id);
                return;
            }
        }
    }
    state.selectedAnnotation = null;
    updateAnnotationList();
    redraw();
}

async function deleteAnnotation(id, event) {
    if (event) event.stopPropagation();
    state.annotations = state.annotations.filter(a => a.id !== id);
    state.selectedAnnotation = null;
    invalidateStaticCache();  // 标注变化，更新缓存
    updateAnnotationList();
    redraw();
    // 自动保存（不标记为手动）
    await saveAnnotations(false, false);
}

function deleteSelectedAnnotation() {
    if (state.selectedAnnotation) {
        deleteAnnotation(state.selectedAnnotation);
    }
}

async function saveAnnotations(showMessage = true, markAsManual = true) {
    if (!state.projectId || state.currentIndex < 0) return;

    try {
        // 如果是手动保存（点击保存按钮或Ctrl+S），将所有标注标记为人工标注
        const annotationsToSave = markAsManual ? state.annotations.map(ann => ({
            ...ann,
            manual: 1  // 标记为手动标注
        })) : state.annotations;

        const response = await fetch('/api/annotation/save', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                project_id: state.projectId,
                image_index: state.currentIndex,
                annotations: annotationsToSave
            })
        });

        const data = await response.json();
        if (data.success) {
            // 更新本地状态
            state.annotations.forEach(ann => {
                if (markAsManual) {
                    ann.manual = 1;
                }
            });

            // 更新图片列表的标注状态
            state.images[state.currentIndex].annotated = state.annotations.length > 0;
            updateImageList();
            if (showMessage) {
                showToast('成功', '标注已保存');
            }
        }
    } catch (error) {
        showToast('错误', error.message, 'danger');
    }
}

// ==================== 类别管理 ====================

function updateClassList() {
    const list = document.getElementById('classList');

    if (state.classes.length === 0) {
        list.innerHTML = '<div class="text-muted small">暂无类别，请先添加</div>';
        state.currentClass = null;
        return;
    }

    // 如果当前没有选中类名，或选中的类名已被删除，自动选中第一个
    if (!state.currentClass || !state.classes.includes(state.currentClass)) {
        state.currentClass = state.classes[0];
    }

    list.innerHTML = state.classes.map((cls, idx) => `
        <div class="class-item ${state.currentClass === cls ? 'selected' : ''}"
             onclick="selectClass('${cls}')" title="点击选择此类别">
            <div class="color-dot" style="background-color: ${colors[idx % colors.length]}"></div>
            <span class="name">${cls}</span>
            <div class="class-actions">
                <i class="bi bi-filter filter-btn" onclick="event.stopPropagation(); filterByClass('${cls}')" title="查看该类别的AI标注"></i>
                <i class="bi bi-trash delete-btn" onclick="event.stopPropagation(); deleteClassAnnotations('${cls}')" title="删除该类别所有标注"></i>
                <i class="bi bi-x remove-btn" onclick="event.stopPropagation(); removeClass('${cls}')" title="移除类别"></i>
            </div>
        </div>
    `).join('');
}

// 选择当前类名
function selectClass(className) {
    state.currentClass = className;
    updateClassList();
    showToast('提示', `已选择类别: ${className}`);
}

async function addClass() {
    const input = document.getElementById('newClassName');
    const name = input.value.trim();

    if (!name || state.classes.includes(name)) return;

    state.classes.push(name);
    updateClassList();
    input.value = '';

    if (state.projectId) {
        await fetch('/api/classes/update', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                project_id: state.projectId,
                classes: state.classes
            })
        });
    }
}

async function removeClass(name) {
    state.classes = state.classes.filter(c => c !== name);
    updateClassList();

    if (state.projectId) {
        await fetch('/api/classes/update', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                project_id: state.projectId,
                classes: state.classes
            })
        });
    }
}

// 删除某个类别的所有标注
async function deleteClassAnnotations(className) {
    if (!state.projectId) return;
    
    if (!confirm(`确定要删除所有 "${className}" 类型的标注吗？\n\n此操作将删除项目中所有该类别的标注，不可恢复！`)) {
        return;
    }
    
    try {
        const response = await fetch('/api/annotation/delete_class', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                project_id: state.projectId,
                class_name: className
            })
        });
        
        const result = await response.json();
        if (result.success) {
            showToast('成功', result.message);
            // 刷新当前图片的标注
            if (state.currentIndex >= 0) {
                await loadImage(state.currentIndex);
            }
        } else {
            showToast('错误', result.error);
        }
    } catch (error) {
        console.error('删除类别标注失败:', error);
        showToast('错误', '删除失败');
    }
}

// 筛选指定类别的AI自动标注（概况模式）
async function filterByClass(className) {
    if (!state.projectId) {
        showToast('错误', '请先打开项目');
        return;
    }
    
    try {
        const response = await fetch('/api/annotation/by_class', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                project_id: state.projectId,
                class_name: className,
                manual_only: false  // 获取所有标注（包括AI和手动）
            })
        });
        
        const result = await response.json();
        if (result.success) {
            // 显示概况信息
            showAnnotationsByClassSummary(className, result.summary);
        } else {
            showToast('错误', result.error);
        }
    } catch (error) {
        console.error('获取类别标注失败:', error);
        showToast('错误', '获取失败');
    }
}

// 显示指定类别的标注概况
function showAnnotationsByClassSummary(className, summary) {
    // 统计总量
    let totalCount = 0, totalAi = 0, totalManual = 0, totalLowConf = 0;
    summary.forEach(img => {
        totalCount += img.annotation_count;
        totalAi += img.ai_count;
        totalManual += img.manual_count;
        totalLowConf += img.low_confidence_count;
    });
    
    // 生成HTML
    let html = `
        <div style="max-height: 70vh; overflow-y: auto;">
            <h5 class="mb-3">类别: ${className}</h5>
            <div class="alert alert-info">
                <strong>总统计:</strong> 
                ${summary.length} 张图片 | 
                总计 ${totalCount} 个标注 | 
                AI自动标注 ${totalAi} 个 | 
                手动标注 ${totalManual} 个 | 
                低置信度（<0.45）${totalLowConf} 个
            </div>
            <div class="mb-3">
                <button class="btn btn-sm btn-outline-primary me-2" onclick="filterByClass('${className}')">
                    <i class="bi bi-arrow-clockwise"></i> 刷新
                </button>
            </div>
    `;
    
    // 按图片显示概况
    summary.forEach(img => {
        const hasLowConf = img.low_confidence_count > 0;
        html += `
            <div class="card mb-2 ${hasLowConf ? 'border-warning' : ''}">
                <div class="card-header py-2 d-flex justify-content-between align-items-center" 
                     style="cursor: pointer;" onclick="event.stopPropagation(); loadImage(${img.image_index}); document.getElementById('customModal').remove();">
                    <small>
                        <i class="bi bi-image"></i> [${img.image_index}] ${img.filename}
                        <span class="badge bg-secondary ms-2">${img.annotation_count}个</span>
                        <span class="badge bg-warning">${img.ai_count} AI</span>
                        <span class="badge bg-success">${img.manual_count} 手动</span>
                        ${hasLowConf ? `<span class="badge bg-danger">${img.low_confidence_count} 低置信</span>` : ''}
                    </small>
                    <i class="bi bi-box-arrow-in-right"></i>
                </div>
                ${img.ai_count > 0 ? `
                <div class="card-body py-1" style="font-size: 11px;">
                    <div class="row g-1">
                        <div class="col-3">
                            <small class="text-muted">平均置信度:</small><br>
                            <strong>${img.avg_ai_score !== null ? img.avg_ai_score.toFixed(3) : '-'}</strong>
                        </div>
                        <div class="col-3">
                            <small class="text-muted">最小:</small><br>
                            <strong>${img.min_ai_score !== null ? img.min_ai_score.toFixed(3) : '-'}</strong>
                        </div>
                        <div class="col-3">
                            <small class="text-muted">最大:</small><br>
                            <strong>${img.max_ai_score !== null ? img.max_ai_score.toFixed(3) : '-'}</strong>
                        </div>
                        <div class="col-3">
                            <small class="text-muted">低置信度:</small><br>
                            <strong class="${img.low_confidence_count > 0 ? 'text-danger' : 'text-success'}">${img.low_confidence_count}个</strong>
                        </div>
                    </div>
                </div>
                ` : ''}
            </div>
        `;
    });
    
    html += '</div>';
    
    // 显示模态框
    showCustomModal(`类别 "${className}" 的标注概况`, html);
}

// 显示类别统计
async function showClassStats() {
    if (!state.projectId) {
        showToast('错误', '请先打开项目');
        return;
    }
    
    try {
        const response = await fetch('/api/annotation/stats_by_class', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                project_id: state.projectId
            })
        });
        
        const result = await response.json();
        if (result.success) {
            showClassStatsModal(result.stats);
        } else {
            showToast('错误', result.error);
        }
    } catch (error) {
        console.error('获取类别统计失败:', error);
        showToast('错误', '获取失败');
    }
}

// 显示类别统计模态框
function showClassStatsModal(stats) {
    let html = `
        <div style="max-height: 70vh; overflow-y: auto;">
            <table class="table table-sm table-bordered table-striped">
                <thead class="table-dark">
                    <tr>
                        <th>类别</th>
                        <th>总计</th>
                        <th>手动</th>
                        <th>AI</th>
                        <th>平均置信度</th>
                        <th>最小</th>
                        <th>最大</th>
                        <th>低置信度</th>
                    </tr>
                </thead>
                <tbody>
    `;
    
    stats.forEach(s => {
        const lowConfCount = s.ai_count - Math.round(s.ai_count * (s.avg_score || 0));
        html += `
            <tr>
                <td><strong>${s.class_name}</strong></td>
                <td>${s.total_count}</td>
                <td class="text-success">${s.manual_count}</td>
                <td class="text-warning">${s.ai_count}</td>
                <td>${s.avg_score !== null ? s.avg_score.toFixed(3) : '-'}</td>
                <td>${s.min_score !== null ? s.min_score.toFixed(3) : '-'}</td>
                <td>${s.max_score !== null ? s.max_score.toFixed(3) : '-'}</td>
                <td class="text-danger">${lowConfCount} (估)</td>
            </tr>
        `;
    });
    
    html += `
                </tbody>
            </table>
        </div>
    `;
    
    showCustomModal('类别标注统计', html);
}

// 清理低置信度标注
async function clearLowConfidenceAnnotations(className = null) {
    if (!state.projectId) {
        showToast('错误', '请先打开项目');
        return;
    }
    
    const threshold = document.getElementById('clearConfidenceSlider').value / 100;
    const classInfo = className ? `类别 "${className}" 的` : '所有';
    
    if (!confirm(`确定要清理 ${classInfo}低置信度标注吗？\n\n此操作将删除置信度低于 ${threshold.toFixed(2)} 的AI自动标注\n保留手动标注和高置信度标注\n此操作不可恢复！`)) {
        return;
    }
    
    try {
        const response = await fetch('/api/annotation/clear_low_confidence', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                project_id: state.projectId,
                confidence_threshold: threshold,
                class_name: className
            })
        });
        
        const result = await response.json();
        if (result.success) {
            showToast('成功', result.message);
            // 刷新当前图片的标注
            if (state.currentIndex >= 0) {
                await loadImage(state.currentIndex);
            }
        } else {
            showToast('错误', result.error);
        }
    } catch (error) {
        console.error('清理低置信度标注失败:', error);
        showToast('错误', '清理失败');
    }
}

// 自定义模态框显示
function showCustomModal(title, content) {
    // 移除已存在的模态框
    const existingModal = document.getElementById('customModal');
    if (existingModal) {
        existingModal.remove();
    }
    
    const modal = document.createElement('div');
    modal.id = 'customModal';
    modal.className = 'modal fade show';
    modal.style.display = 'block';
    modal.innerHTML = `
        <div class="modal-dialog modal-lg modal-dialog-scrollable" onclick="event.stopPropagation()">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title">${title}</h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal"
                            onclick="document.getElementById('customModal').remove()"></button>
                </div>
                <div class="modal-body">
                    ${content}
                </div>
                <div class="modal-footer">
                    <button type="button" class="btn btn-secondary"
                            onclick="document.getElementById('customModal').remove()">关闭</button>
                </div>
            </div>
        </div>
        <div class="modal-backdrop fade show"></div>
    `;

    document.body.appendChild(modal);

    // 只点击背景才关闭，点击模态框内容不会关闭
    const backdrop = modal.querySelector('.modal-backdrop');
    if (backdrop) {
        backdrop.addEventListener('click', (e) => {
            e.stopPropagation();
            modal.remove();
        });
    }
}

// 通过点击删除标注（保留此函数以兼容）
async function deleteAnnotationByClick(imageIndex, annotationId) {
    if (!state.projectId) return;

    if (!confirm('确定要删除此标注吗？')) {
        return;
    }
    
    try {
        const response = await fetch('/api/annotation/delete', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                project_id: state.projectId,
                image_index: imageIndex,
                annotation_id: annotationId
            })
        });
        
        const result = await response.json();
        if (result.success) {
            showToast('成功', '标注已删除');
            // 刷新当前图片
            if (state.currentIndex === imageIndex) {
                await loadImage(imageIndex);
            }
            // 关闭并重新打开模态框
            const modal = document.getElementById('customModal');
            if (modal) {
                modal.remove();
            }
        } else {
            showToast('错误', result.error);
        }
    } catch (error) {
        console.error('删除标注失败:', error);
        showToast('错误', '删除失败');
    }
}

// 删除当前图片
async function deleteCurrentImage() {
    if (!state.projectId || state.currentIndex < 0) return;
    
    const image = state.images[state.currentIndex];
    if (!image) return;
    
    if (!confirm(`确定要删除图片 "${image.filename}" 吗？\n\n此操作将删除该图片及其所有标注，不可恢复！`)) {
        return;
    }
    
    try {
        const response = await fetch('/api/image/delete', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                project_id: state.projectId,
                image_index: state.currentIndex
            })
        });
        
        const result = await response.json();
        if (result.success) {
            showToast('成功', '图片已删除');
            // 重新加载项目
            await selectProject(state.projectId);
            // 如果还有图片，跳转到相同索引或最后一张
            if (state.images.length > 0) {
                const newIndex = Math.min(state.currentIndex, state.images.length - 1);
                await loadImage(newIndex);
            } else {
                // 没有图片了，重置状态
                state.currentIndex = -1;
                state.annotations = [];
                currentImage = null;
                updateAnnotationList();
                updateImageList();
                // 清除画布
                const canvasEl = document.getElementById('annotationCanvas');
                const ctxEl = canvasEl.getContext('2d');
                ctxEl.clearRect(0, 0, canvasEl.width, canvasEl.height);
                document.getElementById('currentImageInfo').textContent = '无图片';
            }
        } else {
            showToast('错误', result.error);
        }
    } catch (error) {
        console.error('删除图片失败:', error);
        showToast('错误', '删除失败');
    }
}

// 清理所有非手动标注
async function clearNonManualAnnotations() {
    if (!state.projectId) return;
    
    if (!confirm(`确定要清理所有非手动标注吗？\n\n此操作将删除项目中所有 AI 自动生成的标注，仅保留手动标注！\n此操作不可恢复！`)) {
        return;
    }
    
    try {
        const response = await fetch('/api/annotation/clear_non_manual', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                project_id: state.projectId
            })
        });
        
        const result = await response.json();
        if (result.success) {
            showToast('成功', result.message);
            // 刷新当前图片的标注
            if (state.currentIndex >= 0) {
                await loadImage(state.currentIndex);
            }
        } else {
            showToast('错误', result.error);
        }
    } catch (error) {
        console.error('清理非手动标注失败:', error);
        showToast('错误', '清理失败');
    }
}

// ==================== 导出 ====================

function showExportModal() {
    if (!state.projectId) {
        showToast('提示', '请先选择项目');
        return;
    }
    new bootstrap.Modal(document.getElementById('exportModal')).show();

    // 自动开始预览
    setTimeout(() => {
        generateExportPreview();
    }, 300);
}

async function exportDataset() {
    const format = document.getElementById('exportFormat').value;
    const outputDir = document.getElementById('exportOutputDir').value.trim();
    const smoothLevel = document.getElementById('exportSmoothLevel').value;
    const exportType = document.getElementById('exportType').value;

    if (!outputDir) {
        showToast('提示', '请输入输出目录');
        return;
    }

    showLoading('正在导出...');

    try {
        const response = await fetch(`/api/export/${format}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                project_id: state.projectId,
                output_dir: outputDir,
                smooth_level: smoothLevel,
                export_type: exportType
            })
        });

        const data = await response.json();
        if (data.success) {
            const result = data.result;
            const smoothNames = {none: '无', low: '低', medium: '中等', high: '高', ultra: '超高'};
            showToast('成功',
                `导出完成!\nTrain: ${result.train}, Val: ${result.val}, Test: ${result.test}\n` +
                `总标注: ${result.total_annotations}\n平滑级别: ${smoothNames[smoothLevel]}`
            );
            bootstrap.Modal.getInstance(document.getElementById('exportModal')).hide();
        }
    } catch (error) {
        showToast('错误', error.message, 'danger');
    }

    hideLoading();
}

// ==================== 导出预览 ====================

async function generateExportPreview() {
    if (!state.projectId) {
        showToast('提示', '请先选择项目');
        return;
    }

    const smoothLevel = document.getElementById('exportSmoothLevel').value;
    const previewImage = document.getElementById('exportPreviewImage');
    const placeholder = document.querySelector('.preview-placeholder');
    const statsDiv = document.getElementById('exportPreviewStats');

    // 显示加载状态
    if (placeholder) placeholder.innerHTML = '<div class="spinner-border spinner-border-sm"></div><p>生成预览中...</p>';

    try {
        const response = await fetch('/api/export/preview', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                project_id: state.projectId,
                image_index: state.currentIndex,
                smooth_level: smoothLevel,
                show_polygon: true,
                show_fill: true,
                opacity: 0.4
            })
        });

        const data = await response.json();
        if (data.success) {
            previewImage.src = data.preview;
            previewImage.style.display = 'block';
            if (placeholder) placeholder.style.display = 'none';

            // 显示统计信息
            const stats = data.stats;
            const smoothNames = {none: '无平滑', low: '低', medium: '中等', high: '高', ultra: '超高'};
            statsDiv.innerHTML = `
                <i class="bi bi-info-circle me-1"></i>
                文件: ${stats.filename} |
                标注数: ${stats.total_annotations} |
                平滑: ${smoothNames[stats.smooth_level]} |
                尺寸: ${stats.image_size[0]}x${stats.image_size[1]}
            `;
            statsDiv.style.display = 'block';
        } else {
            showToast('错误', data.error || '生成预览失败', 'danger');
            if (placeholder) {
                placeholder.innerHTML = '<i class="bi bi-exclamation-triangle"></i><p>预览生成失败</p>';
                placeholder.style.display = 'block';
            }
        }
    } catch (error) {
        showToast('错误', error.message, 'danger');
        if (placeholder) {
            placeholder.innerHTML = '<i class="bi bi-exclamation-triangle"></i><p>预览生成失败</p>';
            placeholder.style.display = 'block';
        }
    }
}

function updateExportPreview() {
    // 如果预览图片已显示，则自动更新预览
    const previewImage = document.getElementById('exportPreviewImage');
    if (previewImage && previewImage.style.display !== 'none') {
        generateExportPreview();
    }
}

async function showSmoothCompare() {
    if (!state.projectId) {
        showToast('提示', '请先选择项目');
        return;
    }

    // 获取当前图片的标注列表
    const annotations = state.annotations || [];
    if (annotations.length === 0) {
        showToast('提示', '当前图片没有标注，无法对比');
        return;
    }

    // 填充标注选择下拉框
    const select = document.getElementById('compareAnnotationSelect');
    select.innerHTML = annotations.map((ann, i) =>
        `<option value="${i}">${i + 1}. ${ann.class_name || ann.label || '未命名'}</option>`
    ).join('');

    // 显示模态框
    new bootstrap.Modal(document.getElementById('smoothCompareModal')).show();

    // 生成对比预览
    await updateSmoothCompare();
}

async function updateSmoothCompare() {
    const annotationIndex = parseInt(document.getElementById('compareAnnotationSelect').value);
    const loadingDiv = document.getElementById('smoothCompareLoading');
    const gridDiv = document.getElementById('smoothCompareGrid');

    // 显示加载状态
    loadingDiv.style.display = 'block';
    gridDiv.style.opacity = '0.3';

    try {
        const response = await fetch('/api/export/preview_compare', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                project_id: state.projectId,
                image_index: state.currentIndex,
                annotation_index: annotationIndex
            })
        });

        const data = await response.json();
        if (data.success) {
            // 更新各级别的预览图片
            const levels = ['none', 'low', 'medium', 'high', 'ultra'];
            levels.forEach(level => {
                const img = document.getElementById(`compare${level.charAt(0).toUpperCase() + level.slice(1)}`);
                if (img && data.previews[level]) {
                    img.src = data.previews[level];
                }
            });

            // 显示原始点数
            document.getElementById('compareNoneInfo').textContent = `原始: ${data.original_points} 点`;
        } else {
            showToast('错误', data.error || '生成对比失败', 'danger');
        }
    } catch (error) {
        showToast('错误', error.message, 'danger');
    }

    // 隐藏加载状态
    loadingDiv.style.display = 'none';
    gridDiv.style.opacity = '1';
}

// ==================== 图片放大查看器 ====================

let viewerZoom = 1;

function openImageViewer(imgSrc, info = '') {
    const viewer = document.getElementById('imageViewer');
    const viewerImg = document.getElementById('imageViewerImg');
    const viewerInfo = document.getElementById('imageViewerInfo');

    viewerImg.src = imgSrc;
    viewerInfo.textContent = info;
    viewerZoom = 1;
    viewerImg.style.transform = `scale(${viewerZoom})`;

    viewer.classList.add('active');
    document.body.style.overflow = 'hidden';
}

function closeImageViewer(event) {
    // 如果点击的是背景或关闭按钮，则关闭
    if (!event || event.target.id === 'imageViewer' || event.target.classList.contains('image-viewer-close')) {
        const viewer = document.getElementById('imageViewer');
        viewer.classList.remove('active');
        document.body.style.overflow = '';
    }
}

function zoomViewerImage(factor) {
    const viewerImg = document.getElementById('imageViewerImg');
    viewerZoom *= factor;
    viewerZoom = Math.max(0.2, Math.min(5, viewerZoom)); // 限制缩放范围
    viewerImg.style.transform = `scale(${viewerZoom})`;
}

function resetViewerZoom() {
    const viewerImg = document.getElementById('imageViewerImg');
    viewerZoom = 1;
    viewerImg.style.transform = `scale(${viewerZoom})`;
}

// 为预览图添加点击事件
function initImageViewerEvents() {
    // 导出预览图
    const exportPreview = document.getElementById('exportPreviewImage');
    if (exportPreview) {
        exportPreview.onclick = function() {
            if (this.src && this.style.display !== 'none') {
                openImageViewer(this.src, '导出预览');
            }
        };
    }

    // 对比预览图
    const compareImgs = document.querySelectorAll('.compare-img');
    compareImgs.forEach(img => {
        img.onclick = function() {
            if (this.src) {
                const levelName = this.id.replace('compare', '');
                openImageViewer(this.src, `平滑级别: ${levelName}`);
            }
        };
    });
}

// ESC 键关闭查看器
document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape') {
        closeImageViewer();
    }
});

// 鼠标滚轮缩放
document.getElementById('imageViewer')?.addEventListener('wheel', function(e) {
    e.preventDefault();
    const factor = e.deltaY > 0 ? 0.9 : 1.1;
    zoomViewerImage(factor);
});

// 页面加载后初始化
document.addEventListener('DOMContentLoaded', function() {
    initImageViewerEvents();
});

// ==================== 缩放控制 ====================

function zoomIn() {
    state.zoom = Math.min(5, state.zoom * 1.2);
    invalidateStaticCache();
    redraw();
    updateZoomDisplay();
}

function zoomOut() {
    state.zoom = Math.max(0.1, state.zoom / 1.2);
    invalidateStaticCache();
    redraw();
    updateZoomDisplay();
}

function resetZoom() {
    state.zoom = 1;
    invalidateStaticCache();
    redraw();
    centerCanvas();
    updateZoomDisplay();
}

// 适应视图 - 让图片完整显示在容器中
function fitToView() {
    if (!currentImage) return;

    const container = document.getElementById('canvasContainer');
    const containerWidth = container.clientWidth - 40; // 留一些边距
    const containerHeight = container.clientHeight - 40;

    const scaleX = containerWidth / currentImage.width;
    const scaleY = containerHeight / currentImage.height;

    state.zoom = Math.min(scaleX, scaleY, 1); // 不超过100%
    invalidateStaticCache();
    redraw();
    centerCanvas();
    updateZoomDisplay();
}

// 将 canvas 居中显示
function centerCanvas() {
    const container = document.getElementById('canvasContainer');
    const wrapper = document.getElementById('canvasWrapper');

    if (!wrapper || !container) return;

    // 计算居中位置
    const scrollX = (wrapper.scrollWidth - container.clientWidth) / 2;
    const scrollY = (wrapper.scrollHeight - container.clientHeight) / 2;

    container.scrollLeft = Math.max(0, scrollX);
    container.scrollTop = Math.max(0, scrollY);
}

// ==================== 工具函数 ====================

function normalizeBox(x1, y1, x2, y2) {
    return {
        x1: Math.min(x1, x2),
        y1: Math.min(y1, y2),
        x2: Math.max(x1, x2),
        y2: Math.max(y1, y2),
        width: Math.abs(x2 - x1),
        height: Math.abs(y2 - y1)
    };
}

function showLoading(text = '处理中...', progress = -1) {
    document.getElementById('loadingText').textContent = text;
    document.getElementById('loadingOverlay').style.display = 'flex';
    updateLoadingProgress(progress);
}

function updateLoadingProgress(progress) {
    const progressBar = document.getElementById('loadingProgress');
    const progressText = document.getElementById('loadingProgressText');

    if (progress < 0) {
        // 不确定进度，显示动画
        progressBar.style.width = '100%';
        progressBar.style.animation = 'loading-pulse 1.5s ease-in-out infinite';
        progressText.textContent = '处理中...';
    } else {
        // 确定进度
        progressBar.style.animation = 'none';
        progressBar.style.width = Math.min(100, Math.max(0, progress)) + '%';
        progressText.textContent = Math.round(progress) + '%';
    }
}

function hideLoading() {
    document.getElementById('loadingOverlay').style.display = 'none';
    // 重置进度
    updateLoadingProgress(0);
}

function showToast(title, message, type = 'success') {
    const toast = document.getElementById('toast');
    document.getElementById('toastTitle').textContent = title;
    document.getElementById('toastBody').textContent = message;

    toast.classList.remove('bg-success', 'bg-danger', 'bg-warning');
    if (type === 'danger') toast.classList.add('bg-danger');

    new bootstrap.Toast(toast).show();
}

// 切换批量选项显示/隐藏
function toggleBatchOptions() {
    const options = document.getElementById('batchOptions');
    const toggle = document.querySelector('.batch-toggle');
    const icon = document.getElementById('batchToggleIcon');

    const isHidden = options.style.display === 'none';
    options.style.display = isHidden ? 'block' : 'none';
    toggle.classList.toggle('expanded', isHidden);
}

// ==================== AI翻译配置 ====================

// AI配置状态
const aiConfig = {
    enabled: false,
    apiUrl: '',
    apiKey: '',
    model: 'deepseek-chat'
};

// 初始化时加载AI配置
function loadAIConfig() {
    const saved = localStorage.getItem('sam3_ai_config');
    if (saved) {
        try {
            const config = JSON.parse(saved);
            Object.assign(aiConfig, config);
            updateAIConfigUI();
        } catch (e) {
            console.error('加载AI配置失败:', e);
        }
    }
}

// 更新AI配置UI状态
function updateAIConfigUI() {
    const btn = document.getElementById('aiConfigBtn');
    const statusText = document.getElementById('aiStatusText');
    const enabledCheckbox = document.getElementById('aiTranslateEnabled');

    if (btn) {
        if (aiConfig.enabled && aiConfig.apiUrl && aiConfig.apiKey) {
            btn.classList.add('ai-enabled');
            btn.title = 'AI翻译已启用';
        } else {
            btn.classList.remove('ai-enabled');
            btn.title = 'AI翻译配置';
        }
    }

    if (statusText) {
        if (aiConfig.enabled && aiConfig.apiUrl && aiConfig.apiKey) {
            statusText.textContent = '已启用';
            statusText.style.color = 'var(--success)';
        } else if (aiConfig.apiUrl && aiConfig.apiKey) {
            statusText.textContent = '已配置但未启用';
            statusText.style.color = 'var(--warning)';
        } else {
            statusText.textContent = '未配置';
            statusText.style.color = 'var(--text-muted)';
        }
    }

    if (enabledCheckbox) {
        enabledCheckbox.checked = aiConfig.enabled;
    }
}

// 显示AI配置模态框
function showAIConfigModal() {
    // 填充当前配置
    document.getElementById('aiApiUrl').value = aiConfig.apiUrl || '';
    document.getElementById('aiApiKey').value = aiConfig.apiKey || '';
    document.getElementById('aiModel').value = aiConfig.model || 'deepseek-chat';
    document.getElementById('aiTranslateEnabled').checked = aiConfig.enabled;

    // 隐藏测试结果
    document.getElementById('aiTestResult').style.display = 'none';

    updateAIConfigUI();
    new bootstrap.Modal(document.getElementById('aiConfigModal')).show();
}

// 切换API密钥可见性
function toggleApiKeyVisibility() {
    const input = document.getElementById('aiApiKey');
    const icon = document.getElementById('apiKeyEyeIcon');

    if (input.type === 'password') {
        input.type = 'text';
        icon.classList.remove('bi-eye');
        icon.classList.add('bi-eye-slash');
    } else {
        input.type = 'password';
        icon.classList.remove('bi-eye-slash');
        icon.classList.add('bi-eye');
    }
}

// 测试AI配置
async function testAIConfig() {
    const apiUrl = document.getElementById('aiApiUrl').value.trim();
    const apiKey = document.getElementById('aiApiKey').value.trim();
    const model = document.getElementById('aiModel').value.trim();

    const resultDiv = document.getElementById('aiTestResult');

    if (!apiUrl || !apiKey) {
        resultDiv.innerHTML = '<div class="alert alert-warning small mb-0"><i class="bi bi-exclamation-triangle"></i> 请填写API地址和密钥</div>';
        resultDiv.style.display = 'block';
        return;
    }

    resultDiv.innerHTML = '<div class="alert alert-info small mb-0"><i class="bi bi-hourglass-split"></i> 正在测试连接...</div>';
    resultDiv.style.display = 'block';

    try {
        const response = await fetch('/api/ai/test', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ api_url: apiUrl, api_key: apiKey, model: model })
        });

        const data = await response.json();

        if (data.success) {
            resultDiv.innerHTML = '<div class="alert alert-success small mb-0"><i class="bi bi-check-circle"></i> 连接成功！API配置有效</div>';
        } else {
            resultDiv.innerHTML = `<div class="alert alert-danger small mb-0"><i class="bi bi-x-circle"></i> 连接失败: ${data.error}</div>`;
        }
    } catch (error) {
        resultDiv.innerHTML = `<div class="alert alert-danger small mb-0"><i class="bi bi-x-circle"></i> 请求错误: ${error.message}</div>`;
    }
}

// 保存AI配置
function saveAIConfig() {
    aiConfig.apiUrl = document.getElementById('aiApiUrl').value.trim();
    aiConfig.apiKey = document.getElementById('aiApiKey').value.trim();
    aiConfig.model = document.getElementById('aiModel').value.trim() || 'deepseek-chat';
    aiConfig.enabled = document.getElementById('aiTranslateEnabled').checked;

    // 如果启用但未配置，提示用户
    if (aiConfig.enabled && (!aiConfig.apiUrl || !aiConfig.apiKey)) {
        showToast('提示', '请先配置API地址和密钥', 'warning');
        return;
    }

    localStorage.setItem('sam3_ai_config', JSON.stringify(aiConfig));
    updateAIConfigUI();

    bootstrap.Modal.getInstance(document.getElementById('aiConfigModal')).hide();

    if (aiConfig.enabled) {
        showToast('成功', 'AI翻译已启用，中文提示词将自动翻译');
    } else {
        showToast('成功', 'AI配置已保存');
    }
}

// 清除AI配置
function clearAIConfig() {
    if (!confirm('确定要清除AI配置吗？')) return;

    aiConfig.enabled = false;
    aiConfig.apiUrl = '';
    aiConfig.apiKey = '';
    aiConfig.model = 'deepseek-chat';
}

// ==================== NMS配置 ====================

// 显示NMS配置模态框
async function showNMSConfigModal() {
    // 加载当前配置
    try {
        const response = await fetch('/api/config/nms');
        const data = await response.json();
        if (data.success) {
            const config = data.config;
            state.nmsConfig = {
                enabled: config.enabled,
                iouThreshold: config.iou_threshold,
                overlapMode: config.overlap_mode,
                minAreaRatio: config.min_area_ratio,
                maskLevel: config.mask_level
            };

            // 更新UI
            document.getElementById('enableNms').checked = state.nmsConfig.enabled;
            document.getElementById('nmsIouSlider').value = state.nmsConfig.iouThreshold * 100;
            document.getElementById('nmsIouValue').textContent = state.nmsConfig.iouThreshold.toFixed(2);
            document.getElementById('nmsOverlapMode').value = state.nmsConfig.overlapMode;
            document.getElementById('nmsMinAreaSlider').value = state.nmsConfig.minAreaRatio * 100;
            document.getElementById('nmsMinAreaValue').textContent = state.nmsConfig.minAreaRatio.toFixed(2);
            document.getElementById('nmsMaskLevel').checked = state.nmsConfig.maskLevel;

            new bootstrap.Modal(document.getElementById('nmsConfigModal')).show();
        }
    } catch (error) {
        console.error('加载NMS配置失败:', error);
        showToast('错误', '加载NMS配置失败', 'danger');
    }
    localStorage.removeItem('sam3_ai_config');

    document.getElementById('aiApiUrl').value = '';
    document.getElementById('aiApiKey').value = '';
    document.getElementById('aiModel').value = 'deepseek-chat';
    document.getElementById('aiTranslateEnabled').checked = false;
    document.getElementById('aiTestResult').style.display = 'none';

    updateAIConfigUI();
    showToast('成功', 'AI配置已清除');
}


// 保存NMS配置
async function saveNMSConfig() {
    try {
        const response = await fetch('/api/config/nms', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                enabled: state.nmsConfig.enabled,
                iou_threshold: state.nmsConfig.iouThreshold,
                overlap_mode: state.nmsConfig.overlapMode,
                min_area_ratio: state.nmsConfig.minAreaRatio,
                mask_level: state.nmsConfig.maskLevel
            })
        });

        const data = await response.json();
        if (data.success) {
            bootstrap.Modal.getInstance(document.getElementById('nmsConfigModal')).hide();
            showToast('成功', 'NMS配置已保存并生效');
        } else {
            showToast('错误', 'NMS配置保存失败', 'danger');
        }
    } catch (error) {
        console.error('保存NMS配置失败:', error);
        showToast('错误', '保存NMS配置失败', 'danger');
    }
}

// 翻译文本（如果启用了AI翻译）
async function translatePrompt(text) {
    // 如果未启用或未配置，直接返回原文
    if (!aiConfig.enabled || !aiConfig.apiUrl || !aiConfig.apiKey) {
        return { success: false, text: text };
    }

    // 检测是否包含中文
    const hasChinese = /[\u4e00-\u9fa5]/.test(text);
    if (!hasChinese) {
        return { success: true, text: text, translated: false };
    }

    try {
        const response = await fetch('/api/ai/translate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                text: text,
                api_url: aiConfig.apiUrl,
                api_key: aiConfig.apiKey,
                model: aiConfig.model
            })
        });

        const data = await response.json();

        if (data.success) {
            console.log(`[AI翻译] "${text}" -> "${data.translated}"`);
            return { success: true, text: data.translated, original: text, translated: true };
        } else {
            console.warn('[AI翻译失败]', data.error);
            return { success: false, text: text, error: data.error };
        }
    } catch (error) {
        console.error('[AI翻译错误]', error);
        return { success: false, text: text, error: error.message };
    }
}

// 在页面加载时初始化AI配置
document.addEventListener('DOMContentLoaded', () => {
    loadAIConfig();
});
