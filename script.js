// Visualization data from plys folder
// cameraOrbit format: "horizontal vertical distance" (e.g., "45deg 75deg auto")
// horizontal: 水平角度 (0deg = front, 90deg = right, 180deg = back, 270deg = left)
// vertical: 垂直角度 (0deg = top, 90deg = side, 180deg = bottom)
// distance: 距离 (auto or percentage like 120%)
const visualizationScenes = [
    {
        name: "Sequence Scene 1",
        folder: "glbs/1",
        glb: "glbs/1/meeting_room.glb",  // 如果有GLB文件
        images: ["000.png", "001.png", "002.png", "003.png"],
        cameraOrbit: "149deg 64deg 4m"
    },
    {
        name: "Sequence Scene 2",
        folder: "glbs/2",
        glb: "glbs/2/hospital.glb",
        images: ["000.png", "001.png", "003.png", "004.png"],
        cameraOrbit: "182deg 87deg 4m"
    },
    {
        name: "Sequence Scene 3",
        folder: "glbs/3",
        glb: "glbs/3/living_room.glb",
        images: ["000.png", "001.png", "002.png"],
        cameraOrbit: "192deg 87deg 4m"
    },
    {
        name: "Sequence Scene 4",
        folder: "glbs/4",
        glb: "glbs/4/beverage.glb",
        images: ["000.png", "001.png", "002.png"],
        cameraOrbit: "200deg 100deg 4m"
    },
    {
        name: "Sequence Scene 5",
        folder: "glbs/5",
        glb: "glbs/5/massage_room.glb",
        images: ["000.png", "001.png", "002.png", "003.png", "004.png", "005.png", "006.png"],
        cameraOrbit: "224deg 80deg 4m"
    },
    {
        name: "Surround-view Scene 1",
        folder: "glbs/6",
        glb: "glbs/6/buildings.glb",
        images: ["000.jpg", "001.jpg", "002.jpg"],
        cameraOrbit: "177deg 106deg 4m"
    },
    {
        name: "Surround-view Scene 2",
        folder: "glbs/7",
        glb: "glbs/7/village.glb",  // 使用PLY文件
        images: ["005.jpg", "001.jpg", "002.jpg", "003.jpg", "004.jpg"],
        cameraOrbit: "170deg 100deg 15m"
    },
    {
        name: "Surround-view Scene 3",
        folder: "glbs/8",
        glb: "glbs/8/bread.glb",  // 使用PLY文件
        images: ["000.jpg", "001.jpg",],
        cameraOrbit: "-170deg 107deg 4m"
    },
    {
        name: "Surround-view Scene 4",
        folder: "glbs/9",
        glb: "glbs/9/tunnel.glb",  // 使用PLY文件
        images: ["000.jpg", "001.jpg", "002.jpg", "003.jpg",],
        cameraOrbit: "178deg 95deg 4m"
    },
    {
        name: "Monocular Scene 1",
        folder: "glbs/10",
        glb: "glbs/10/barn.glb",  // 使用PLY文件
        images: ["000.jpg", "001.jpg", "002.jpg", "003.jpg",],
        cameraOrbit: "217deg 68deg 15m"
    },
    {
        name: "Monocular Scene 2",
        folder: "glbs/11",
        glb: "glbs/11/toy.glb",  // 使用PLY文件
        images: ["000.jpg",],
        cameraOrbit: "178deg 95deg 10m"
    },
];

let currentSceneIndex = 0;
let modelViewer;
let thumbnailIntervals = {}; // 存储每个缩略图的播放间隔
let glbCache = {}; // 缓存转换后的GLB模型
let isInitialLoad = true; // 标记是否为初始加载

// 确保页面始终从顶部开始
if (history.scrollRestoration) {
    history.scrollRestoration = 'manual';
}

// Initialize the visualization on page load
document.addEventListener('DOMContentLoaded', function () {
    // 强制滚动到顶部
    window.scrollTo(0, 0);

    modelViewer = document.getElementById('modelViewer');
    if (modelViewer) initializeVisualization();

    // 初始化完成后重置标志
    setTimeout(() => {
        isInitialLoad = false;
    }, 100);
});

// Load 3D model using model-viewer
function loadModel(sceneData) {
    const loadingIndicator = document.getElementById('loadingIndicator');

    // Show loading indicator
    if (loadingIndicator) {
        loadingIndicator.style.display = 'flex';
        loadingIndicator.innerHTML = '<div class="loading-spinner"></div><div>Loading 3D Model...</div>';
    }

    // Check if GLB exists, otherwise convert PLY
    if (sceneData.glb) {
        // Direct GLB loading
        modelViewer.src = sceneData.glb;
        applyCameraOrbitSettings(sceneData);

        // Hide loading indicator when model loads and apply settings again
        modelViewer.addEventListener('load', function () {
            // 模型加载完成后再次应用设置，确保orientation生效
            applyCameraOrbitSettings(sceneData);
            if (loadingIndicator) loadingIndicator.style.display = 'none';
        }, { once: true });

        // Handle load error
        modelViewer.addEventListener('error', function (event) {
            console.error('Error loading model:', event);
            if (loadingIndicator) loadingIndicator.innerHTML = '<div style="color: #ff6b6b;">Error loading model</div>';
        }, { once: true });
    } else if (sceneData.ply) {
        // Convert PLY to GLB
        const cacheKey = sceneData.ply;

        if (glbCache[cacheKey]) {
            // Use cached GLB
            console.log('Using cached GLB for:', sceneData.name);
            modelViewer.src = glbCache[cacheKey];
            applyCameraOrbitSettings(sceneData);

            // 等待模型加载完成后再次应用设置
            modelViewer.addEventListener('load', function () {
                applyCameraOrbitSettings(sceneData);
                if (loadingIndicator) loadingIndicator.style.display = 'none';
            }, { once: true });
        } else {
            // Convert PLY to GLB
            if (loadingIndicator) loadingIndicator.innerHTML = '<div class="loading-spinner"></div><div>Converting PLY to GLB...</div>';
            convertPLYtoGLB(sceneData.ply, function (glbUrl) {
                glbCache[cacheKey] = glbUrl;
                modelViewer.src = glbUrl;
                applyCameraOrbitSettings(sceneData);

                // 等待模型加载完成后再次应用设置
                modelViewer.addEventListener('load', function () {
                    applyCameraOrbitSettings(sceneData);
                    if (loadingIndicator) loadingIndicator.style.display = 'none';
                }, { once: true });
            });
        }
    } else {
        console.error('No GLB or PLY file specified for:', sceneData.name);
        if (loadingIndicator) loadingIndicator.innerHTML = '<div style="color: #ff6b6b;">No 3D model available</div>';
    }
}

// Apply camera orbit settings
function applyCameraOrbitSettings(sceneData) {
    const orbit = sceneData.cameraOrbit || "180deg 70deg auto";

    // 确保旋转限制允许完全自由的旋转
    modelViewer.minCameraOrbit = "auto 0deg auto";  // 允许任意距离和水平角度，垂直0-180度
    modelViewer.maxCameraOrbit = "auto 180deg auto"; // 允许360度水平旋转

    // 应用模型旋转对齐坐标轴（如果指定）
    if (sceneData.orientation) {
        modelViewer.orientation = sceneData.orientation;
        console.log('Applied model orientation:', sceneData.orientation);
    } else {
        // 默认不旋转
        modelViewer.orientation = "0deg 0deg 0deg";
    }

    modelViewer.cameraOrbit = orbit;
    modelViewer.resetTurntableRotation(0);
    modelViewer.jumpCameraToGoal();
}

// Convert PLY to GLB for model-viewer
function convertPLYtoGLB(plyPath, callback) {
    console.log('🔄 Converting PLY to GLB:', plyPath);

    const loader = new THREE.PLYLoader();
    loader.load(
        plyPath,
        function (geometry) {
            console.log('✅ PLY loaded, vertices:', geometry.attributes.position.count);
            geometry.computeVertexNormals();

            // Create point cloud mesh with appropriate size
            const material = new THREE.PointsMaterial({
                size: 0.01,  // Point size
                vertexColors: true,
                sizeAttenuation: true
            });
            const points = new THREE.Points(geometry, material);

            // Center and scale the point cloud
            geometry.computeBoundingBox();
            const center = geometry.boundingBox.getCenter(new THREE.Vector3());
            points.position.sub(center);

            const size = geometry.boundingBox.getSize(new THREE.Vector3());
            const maxDim = Math.max(size.x, size.y, size.z);
            const scale = 2.0 / maxDim;
            points.scale.set(scale, scale, scale);

            // Create scene for export
            const tempScene = new THREE.Scene();
            tempScene.add(points);

            // Export to GLB
            const exporter = new THREE.GLTFExporter();
            exporter.parse(
                tempScene,
                function (gltf) {
                    const blob = new Blob([gltf], { type: 'model/gltf-binary' });
                    const url = URL.createObjectURL(blob);
                    console.log('✅ PLY successfully converted to GLB');
                    callback(url);
                },
                function (error) {
                    console.error('❌ Error exporting GLB:', error);
                },
                { binary: true }
            );
        },
        function (xhr) {
            const percentComplete = Math.round((xhr.loaded / xhr.total) * 100);
            console.log('📥 Loading PLY: ' + percentComplete + '%');
        },
        function (error) {
            console.error('❌ Error loading PLY:', error);
        }
    );
}

function initializeVisualization() {
    const thumbnailTrack = document.getElementById('thumbnailTrack');
    if (!thumbnailTrack) return;

    // Generate thumbnails using first image of each scene
    visualizationScenes.forEach((sceneData, index) => {
        const thumbnailItem = document.createElement('div');
        thumbnailItem.className = 'thumbnail-item' + (index === 0 ? ' active' : '');
        thumbnailItem.onclick = () => selectScene(index);
        thumbnailItem.dataset.sceneIndex = index;

        const firstImage = `${sceneData.folder}/${sceneData.images[0]}`;
        const hasMultipleImages = sceneData.images.length > 1;
        thumbnailItem.innerHTML = `
            <img src="${firstImage}" alt="${sceneData.name}" class="thumbnail-image">
            <div class="thumbnail-label">${sceneData.name}</div>
            <div class="thumbnail-progress"></div>
            ${hasMultipleImages ? '<div class="thumbnail-play-icon">▶</div>' : ''}
        `;

        // Add hover event to play image sequence in the thumbnail itself
        thumbnailItem.addEventListener('mouseenter', () => {
            playThumbnailSequence(thumbnailItem, index);
        });

        thumbnailItem.addEventListener('mouseleave', () => {
            stopThumbnailSequence(thumbnailItem, index);
        });

        thumbnailTrack.appendChild(thumbnailItem);
    });

    // Load first scene
    selectScene(0);
}

function selectScene(index) {
    currentSceneIndex = index;
    const sceneData = visualizationScenes[index];

    // Update active thumbnail
    document.querySelectorAll('.thumbnail-item').forEach((item, i) => {
        item.classList.toggle('active', i === index);
    });

    // Update viewer info
    const viewerInfo = document.getElementById('viewerInfo');
    const viewerCounter = document.getElementById('viewerCounter');
    if (viewerInfo) viewerInfo.textContent = sceneData.name;
    if (viewerCounter) viewerCounter.textContent = `${index + 1} / ${visualizationScenes.length}`;

    // Load 3D model with custom camera settings
    if (modelViewer) loadModel(sceneData);

    // Scroll thumbnail into view (但初始加载时不滚动，避免页面跳转)
    if (!isInitialLoad) {
        const thumbnailTrack = document.getElementById('thumbnailTrack');
        const activeThumbnail = thumbnailTrack.children[index];
        if (activeThumbnail) {
            activeThumbnail.scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'center' });
        }
    }
}

// Play image sequence in the thumbnail itself
function playThumbnailSequence(thumbnailItem, sceneIndex) {
    const sceneData = visualizationScenes[sceneIndex];
    if (sceneData.images.length <= 1) return; // No sequence to play

    const img = thumbnailItem.querySelector('.thumbnail-image');
    const progressBar = thumbnailItem.querySelector('.thumbnail-progress');
    let currentIndex = 0;

    // Update image immediately
    if (img) img.src = `${sceneData.folder}/${sceneData.images[currentIndex]}`;
    if (progressBar) progressBar.style.width = '0%';

    // Start playing sequence
    thumbnailIntervals[sceneIndex] = setInterval(() => {
        currentIndex = (currentIndex + 1) % sceneData.images.length;
        if (img) img.src = `${sceneData.folder}/${sceneData.images[currentIndex]}`;

        // Update progress bar
        const progress = ((currentIndex + 1) / sceneData.images.length) * 100;
        if (progressBar) progressBar.style.width = progress + '%';
    }, 300); // Change image every 300ms
}

// Stop playing sequence and reset to first image
function stopThumbnailSequence(thumbnailItem, sceneIndex) {
    if (thumbnailIntervals[sceneIndex]) {
        clearInterval(thumbnailIntervals[sceneIndex]);
        delete thumbnailIntervals[sceneIndex];
    }

    const sceneData = visualizationScenes[sceneIndex];
    const img = thumbnailItem.querySelector('.thumbnail-image');
    const progressBar = thumbnailItem.querySelector('.thumbnail-progress');

    // Reset to first image
    if (img) img.src = `${sceneData.folder}/${sceneData.images[0]}`;
    if (progressBar) progressBar.style.width = '0%';
}

function scrollCarousel(direction) {
    const thumbnailTrack = document.getElementById('thumbnailTrack');
    const scrollAmount = 200;
    if (thumbnailTrack) thumbnailTrack.scrollBy({ left: direction * scrollAmount, behavior: 'smooth' });
}

// Keyboard navigation
document.addEventListener('keydown', function (e) {
    // 防止在输入框中触发
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
        return;
    }

    if (e.key === 'ArrowLeft') {
        const prevIndex = (currentSceneIndex - 1 + visualizationScenes.length) % visualizationScenes.length;
        selectScene(prevIndex);
    } else if (e.key === 'ArrowRight') {
        const nextIndex = (currentSceneIndex + 1) % visualizationScenes.length;
        selectScene(nextIndex);
    }
});

// Ablation Scene Data
const ablationScenes = [
    {
        name: "Sequence Ablation 1",
        type: "camera",
        folder: "glbs/ablations/camera/0",
        images: ["DSC_0286.JPG", "DSC_0296.JPG", "DSC_0299.JPG", "DSC_0314.JPG"],
        modelWithout: "glbs/ablations/camera/0/scene_0_woc.glb",
        modelWith: "glbs/ablations/camera/0/scene_0_wc.glb",
        cameraOrbit: "228deg 79deg 5m",
        cameraTarget: "0.08m 0.64m -0.05m"
    },
    {
        name: "Sequence Ablation 2",
        type: "camera",
        folder: "glbs/ablations/camera/1",
        images: ["DSC_0676.JPG", "DSC_0680.JPG", "DSC_0704.JPG"],
        modelWithout: "glbs/ablations/camera/1/scene_19_woc.glb",
        modelWith: "glbs/ablations/camera/1/scene_19_wc.glb",
        cameraOrbit: "260deg 77deg 5m",
        cameraTarget: "0.26m 0.09m -0.56m"
    },
    {
        name: "Surround-view Ablation 1",
        type: "depth",
        folder: "glbs/ablations/depth/0",
        images: ["Image_16_0_0001_0.png", "Image_37_0_0001_0.png", "Image_47_0_0001_0.png", "Image_97_0_0001_0.png"],
        modelWithout: "glbs/ablations/depth/0/scene_7_wod.glb",
        modelWith: "glbs/ablations/depth/0/scene_7_wd.glb",
        cameraOrbit: "169deg 78deg 2m",
        cameraTarget: "0.43m 0.09m 0.31m"
    },
    {
        name: "Surround-view Ablation 2",
        type: "depth",
        folder: "glbs/ablations/depth/1",
        images: ["Image_26_0_0001_0.png", "Image_33_0_0001_0.png", "Image_81_0_0001_0.png", "Image_91_0_0001_0.png"],
        modelWithout: "glbs/ablations/depth/1/scene_25_wod.glb",
        modelWith: "glbs/ablations/depth/1/scene_25_wd.glb",
        cameraOrbit: "184deg 88deg 2m",
        cameraTarget: "-0.06m -0.04m 0.27m"
    }
];

let currentAblationIndex = 0;

// Initialize ablation scene on page load
document.addEventListener('DOMContentLoaded', function () {
    const ablationCounter = document.getElementById('ablationCounter');
    if (ablationCounter) loadAblationScene(0);
});

// Change ablation scene
function changeAblationScene(direction) {
    currentAblationIndex += direction;

    // Wrap around
    if (currentAblationIndex < 0) {
        currentAblationIndex = ablationScenes.length - 1;
    } else if (currentAblationIndex >= ablationScenes.length) {
        currentAblationIndex = 0;
    }

    loadAblationScene(currentAblationIndex);
}

// Load ablation scene
function loadAblationScene(index) {
    const scene = ablationScenes[index];

    // Update counter
    const ablationCounter = document.getElementById('ablationCounter');
    if (ablationCounter) ablationCounter.textContent = `Scene ${index + 1} / ${ablationScenes.length}`;

    // Update labels based on scene type
    const leftLabel = document.getElementById('ablationLeftLabel');
    const rightLabel = document.getElementById('ablationRightLabel');

    if (leftLabel && rightLabel) {
        if (scene.type === 'camera') {
            leftLabel.textContent = 'w/o Segmentation Forcing';
            rightLabel.textContent = 'w/ Segmentation Forcing';
        } else if (scene.type === 'depth') {
            leftLabel.textContent = 'w/o Novel View Rendering';
            rightLabel.textContent = 'w/ Novel View Rendering';
        }
    }

    // Update images
    const imageGrid = document.getElementById('ablationImageGrid');
    if (imageGrid) {
        imageGrid.innerHTML = '';
        scene.images.forEach((img, i) => {
            const imgElement = document.createElement('img');
            imgElement.src = `${scene.folder}/${img}`;
            imgElement.alt = `Input ${i + 1}`;
            imageGrid.appendChild(imgElement);
        });
    }

    // Update models
    const leftViewer = document.getElementById('ablationModelLeft');
    const rightViewer = document.getElementById('ablationModelRight');

    if (leftViewer && rightViewer) {
        // Function to apply camera settings
        const applyCameraSettings = (viewer) => {
            // Set camera orbit restrictions
            viewer.minCameraOrbit = "auto 0deg 1m";
            viewer.maxCameraOrbit = "auto 180deg 10m";

            // Apply scene-specific camera settings
            viewer.cameraOrbit = scene.cameraOrbit;
            if (scene.cameraTarget) {
                viewer.cameraTarget = scene.cameraTarget;
            }
            viewer.jumpCameraToGoal();
        };

        // Set up event listeners to apply settings after model loads
        const leftLoadHandler = () => {
            applyCameraSettings(leftViewer);
        };
        const rightLoadHandler = () => {
            applyCameraSettings(rightViewer);
        };

        // Remove old listeners if any
        leftViewer.removeEventListener('load', leftLoadHandler);
        rightViewer.removeEventListener('load', rightLoadHandler);

        // Add new listeners
        leftViewer.addEventListener('load', leftLoadHandler, { once: true });
        rightViewer.addEventListener('load', rightLoadHandler, { once: true });

        // Update model sources
        leftViewer.src = scene.modelWithout;
        rightViewer.src = scene.modelWith;

        // Also apply immediately in case models are already loaded
        applyCameraSettings(leftViewer);
        applyCameraSettings(rightViewer);
    }

    // Reset slider position to center
    const slider = document.getElementById('comparisonSlider');
    const leftModel = document.querySelector('.left-model');
    const rightModel = document.querySelector('.right-model');

    if (slider && leftModel && rightModel) {
        slider.style.left = '50%';
        leftModel.style.clipPath = 'inset(0 50% 0 0)';
        rightModel.style.clipPath = 'inset(0 0 0 50%)';
    }

}

// Comparison Slider Functionality
(function () {
    const slider = document.getElementById('comparisonSlider');
    const leftModel = document.querySelector('.left-model');
    const rightModel = document.querySelector('.right-model');
    const container = document.querySelector('.comparison-viewer-container');
    const leftViewer = document.getElementById('ablationModelLeft');
    const rightViewer = document.getElementById('ablationModelRight');

    if (!slider || !container) return;

    let isDragging = false;

    function updateSliderPosition(clientX) {
        const rect = container.getBoundingClientRect();
        let position = ((clientX - rect.left) / rect.width) * 100;
        position = Math.max(0, Math.min(100, position));

        slider.style.left = position + '%';
        leftModel.style.clipPath = `inset(0 ${100 - position}% 0 0)`;
        rightModel.style.clipPath = `inset(0 0 0 ${position}%)`;
    }

    slider.addEventListener('mousedown', (e) => {
        isDragging = true;
        e.preventDefault();
    });

    document.addEventListener('mousemove', (e) => {
        if (!isDragging) return;
        updateSliderPosition(e.clientX);
    });

    document.addEventListener('mouseup', () => {
        isDragging = false;
    });

    // Touch events for mobile
    slider.addEventListener('touchstart', (e) => {
        isDragging = true;
        e.preventDefault();
    });

    document.addEventListener('touchmove', (e) => {
        if (!isDragging) return;
        updateSliderPosition(e.touches[0].clientX);
    });

    document.addEventListener('touchend', () => {
        isDragging = false;
    });

    // Synchronize camera movements
    function syncCameraOrbit() {
        if (leftViewer && rightViewer) {
            const leftOrbit = leftViewer.getCameraOrbit();
            rightViewer.cameraOrbit = `${leftOrbit.theta}rad ${leftOrbit.phi}rad ${leftOrbit.radius}m`;
        }
    }

    function syncCameraTarget() {
        if (leftViewer && rightViewer) {
            const leftTarget = leftViewer.getCameraTarget();
            rightViewer.cameraTarget = `${leftTarget.x}m ${leftTarget.y}m ${leftTarget.z}m`;
        }
    }

    // Add camera sync listeners
    if (leftViewer && rightViewer) {
        leftViewer.addEventListener('camera-change', () => {
            syncCameraOrbit();
            syncCameraTarget();
        });

        rightViewer.addEventListener('camera-change', () => {
            const rightOrbit = rightViewer.getCameraOrbit();
            leftViewer.cameraOrbit = `${rightOrbit.theta}rad ${rightOrbit.phi}rad ${rightOrbit.radius}m`;

            const rightTarget = rightViewer.getCameraTarget();
            leftViewer.cameraTarget = `${rightTarget.x}m ${rightTarget.y}m ${rightTarget.z}m`;
        });
    }
})();

// Comparison Visualization Interactive Controls
let currentVLAPage = 0;
const scenesPerPage = 2;

const vlaScenes = [
    {
        name: 'Sequence Qualitative Example 1',
        baseline: { src: 'vla/kosmos/0-2-lift_red_block_slider-fail%20(1).gif', label: 'Baseline' },
        occany: { src: 'vla/freefusion/0-2-lift_red_block_slider-succ.gif', label: '<span class="gradient-text">OccAny</span>' }
    },
    {
        name: 'Sequence Qualitative Example 2',
        baseline: { src: 'vla/kosmos/16-2-stack_block-fail.gif', label: 'Baseline' },
        occany: { src: 'vla/freefusion/16-2-stack_block-succ.gif', label: '<span class="gradient-text">OccAny</span>' }
    },
    {
        name: 'Ablation Example (Segmentation Forcing)',
        baseline: { src: 'vla/kosmos/3-2-place_in_slider-fail.gif', label: 'w/o Forcing' },
        occany: { src: 'vla/freefusion/3-2-place_in_slider-succ.gif', label: 'w/ Forcing' }
    },
    {
        name: 'Ablation Example (Novel View Rendering)',
        baseline: { src: 'vla/kosmos/5-0-push_blue_block_left-fail.gif', label: 'w/o NVR' },
        occany: { src: 'vla/freefusion/5-0-push_blue_block_left-succ.gif', label: 'w/ NVR' }
    }
];

// Calculate total pages
const totalVLAPages = Math.ceil(vlaScenes.length / scenesPerPage);

// Initialize VLA visualization
function initializeVLA() {
    loadVLAPage(currentVLAPage);
    initializeVLASliders();
}

// Change VLA page
function changeVLAScene(direction) {
    const newPage = currentVLAPage + direction;

    if (newPage < 0 || newPage >= totalVLAPages) {
        return; // Don't go beyond bounds
    }

    currentVLAPage = newPage;
    loadVLAPage(currentVLAPage);
}

// Load VLA page (showing 2 scenes)
function loadVLAPage(pageIndex) {
    if (pageIndex < 0 || pageIndex >= totalVLAPages) return;

    // Update counter
    const counter = document.getElementById('vlaCounter');
    if (counter) {
        counter.textContent = `${pageIndex + 1} / ${totalVLAPages}`;
    }

    // Update navigation buttons
    const prevBtn = document.getElementById('vlaPrevBtn');
    const nextBtn = document.getElementById('vlaNextBtn');
    if (prevBtn) prevBtn.disabled = pageIndex === 0;
    if (nextBtn) nextBtn.disabled = pageIndex === totalVLAPages - 1;

    // Load scene 1
    const scene1Index = pageIndex * scenesPerPage;
    if (scene1Index < vlaScenes.length) {
        const scene1 = vlaScenes[scene1Index];
        updateScene(1, scene1);
    } else {
        // Hide scene 1 if not available
        const scene1Container = document.querySelector('.vla-scenes-container > div:first-child');
        if (scene1Container) scene1Container.style.display = 'none';
    }

    // Load scene 2
    const scene2Index = pageIndex * scenesPerPage + 1;
    if (scene2Index < vlaScenes.length) {
        const scene2 = vlaScenes[scene2Index];
        updateScene(2, scene2);
    } else {
        // Hide scene 2 if not available
        const scene2Container = document.querySelector('.vla-scenes-container > div:last-child');
        if (scene2Container) scene2Container.style.display = 'none';
    }
}

// Update a single scene
function updateScene(sceneNum, scene) {
    const viewerTitle = document.getElementById(`vlaViewerTitle${sceneNum}`);
    const leftGif = document.getElementById(`vlaLeftGif${sceneNum}`);
    const rightGif = document.getElementById(`vlaRightGif${sceneNum}`);
    const leftLabel = document.getElementById(`vlaLeftLabel${sceneNum}`);
    const rightLabel = document.getElementById(`vlaRightLabel${sceneNum}`);

    if (viewerTitle) {
        viewerTitle.textContent = scene.name;
    }
    if (leftGif) {
        leftGif.src = scene.baseline.src;
        leftGif.alt = scene.baseline.label;
    }
    if (rightGif) {
        rightGif.src = scene.occany.src;
        rightGif.alt = scene.occany.label;
    }
    if (leftLabel) {
        leftLabel.innerHTML = scene.baseline.label;
    }
    if (rightLabel) {
        rightLabel.innerHTML = scene.occany.label;
    }

    // Show the scene container
    const sceneContainer = document.querySelector(`.vla-scenes-container > div:nth-child(${sceneNum})`);
    if (sceneContainer) sceneContainer.style.display = 'flex';

    // Reset slider position to center
    const slider = document.getElementById(`vlaSlider${sceneNum}`);
    if (slider && leftGif && rightGif) {
        slider.style.left = '50%';
        leftGif.style.clipPath = 'inset(0 50% 0 0)';
        rightGif.style.clipPath = 'inset(0 0 0 50%)';
    }
}

// Initialize VLA sliders for all scenes
function initializeVLASliders() {
    [1, 2].forEach(sceneNum => {
        initializeVLASlider(sceneNum);
    });
}

// Initialize slider for a specific scene
function initializeVLASlider(sceneNum) {
    const slider = document.getElementById(`vlaSlider${sceneNum}`);
    const leftGif = document.getElementById(`vlaLeftGif${sceneNum}`);
    const rightGif = document.getElementById(`vlaRightGif${sceneNum}`);
    const container = document.getElementById(`vlaContainer${sceneNum}`);

    if (!slider || !container) return;

    let isDragging = false;

    function updateSliderPosition(clientX) {
        const rect = container.getBoundingClientRect();
        let position = ((clientX - rect.left) / rect.width) * 100;
        position = Math.max(0, Math.min(100, position));

        slider.style.left = position + '%';
        if (leftGif) leftGif.style.clipPath = `inset(0 ${100 - position}% 0 0)`;
        if (rightGif) rightGif.style.clipPath = `inset(0 0 0 ${position}%)`;
    }

    slider.addEventListener('mousedown', (e) => {
        isDragging = true;
        e.preventDefault();
    });

    document.addEventListener('mousemove', (e) => {
        if (!isDragging) return;
        updateSliderPosition(e.clientX);
    });

    document.addEventListener('mouseup', () => {
        isDragging = false;
    });

    // Touch events for mobile
    slider.addEventListener('touchstart', (e) => {
        isDragging = true;
        e.preventDefault();
    });

    document.addEventListener('touchmove', (e) => {
        if (!isDragging) return;
        updateSliderPosition(e.touches[0].clientX);
    });

    document.addEventListener('touchend', () => {
        isDragging = false;
    });
}


// Initialize VLA on page load
document.addEventListener('DOMContentLoaded', function () {
    const vlaCounter = document.getElementById('vlaCounter');
    if (vlaCounter) initializeVLA();
});

// Keyboard navigation for VLA
document.addEventListener('keydown', function (e) {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
        return;
    }

    if (e.key === 'ArrowLeft') {
        const vlaCounter = document.getElementById('vlaCounter');
        if (vlaCounter) changeVLAScene(-1);
    } else if (e.key === 'ArrowRight') {
        const vlaCounter = document.getElementById('vlaCounter');
        if (vlaCounter) changeVLAScene(1);
    }
});

// Video Modal Functionality
document.addEventListener('DOMContentLoaded', function() {
    const videoModal = document.getElementById('videoModal');
    const modalVideo = document.getElementById('modalVideo');
    const modalVideoTag = document.getElementById('modalVideoTag');
    const closeModal = document.getElementById('closeModal');
    const teaserContainers = document.querySelectorAll('.video-tag-container');

    if (!videoModal || !modalVideo || !closeModal) return;

    teaserContainers.forEach(container => {
        container.style.cursor = 'pointer';
        container.addEventListener('click', function() {
            const video = this.querySelector('video');
            const source = video.querySelector('source').src;
            const tagText = this.querySelector('.video-tag').textContent;
            
            // Use absolute URL to avoid potential issues
            const absoluteSource = new URL(source, window.location.href).href;
            
            modalVideo.querySelector('source').src = absoluteSource;
            modalVideo.load();
            
            // Set tag text
            if (modalVideoTag) modalVideoTag.textContent = tagText;
            
            videoModal.style.display = 'flex';
            // Trigger reflow for animation
            videoModal.offsetHeight;
            videoModal.classList.add('show');
            
            // Unmute for big player
            modalVideo.muted = false; 
        });
    });

    function closeVideoModal() {
        videoModal.classList.remove('show');
        setTimeout(() => {
            videoModal.style.display = 'none';
            modalVideo.pause();
            modalVideo.querySelector('source').src = '';
            if (modalVideoTag) modalVideoTag.textContent = '';
        }, 400); // Match CSS transition time
    }

    closeModal.addEventListener('click', closeVideoModal);
    
    // Close on background click
    videoModal.addEventListener('click', function(e) {
        if (e.target === videoModal) {
            closeVideoModal();
        }
    });

    // Close on escape key
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape' && videoModal.classList.contains('show')) {
            closeVideoModal();
        }
    });
});

// Video Loading Progress
document.addEventListener('DOMContentLoaded', function() {
    const videos = document.querySelectorAll('video');
    
    videos.forEach(video => {
        // Create wrapper and loading indicator
        const wrapper = document.createElement('div');
        wrapper.style.position = 'relative';
        wrapper.style.width = '100%';
        wrapper.style.display = 'flex';
        // Inherit border radius
        wrapper.style.borderRadius = window.getComputedStyle(video).borderRadius || 'inherit';
        
        // Insert wrapper before video, then move video inside
        video.parentNode.insertBefore(wrapper, video);
        wrapper.appendChild(video);
        
        const loader = document.createElement('div');
        loader.className = 'video-loading-indicator';
        loader.innerHTML = '<div class="loading-spinner"></div>';
        wrapper.appendChild(loader);
        
        const updateLoader = () => {
            if (video.readyState >= 3) {
                loader.style.display = 'none';
            } else {
                loader.style.display = 'flex';
            }
        };
        
        // Add event listeners
        video.addEventListener('waiting', () => loader.style.display = 'flex');
        video.addEventListener('playing', () => loader.style.display = 'none');
        video.addEventListener('canplay', updateLoader);
        video.addEventListener('loadeddata', updateLoader);
        video.addEventListener('loadstart', () => loader.style.display = 'flex');
        
        // Initial state
        updateLoader();
    });
});
