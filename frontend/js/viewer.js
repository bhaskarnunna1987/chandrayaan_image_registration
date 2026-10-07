/**
 * viewer.js
 * Multi-mode registration results inspector:
 * - Interactive Before/After Split-Screen Slider
 * - Alpha Blend Overlay with dynamic opacity control
 * - Dynamic Checkerboard Pattern with configurable tile size
 * - False-color Residual Difference Heatmap
 */

class RegistrationViewer {
  constructor(containerElement) {
    this.container = containerElement;
    this.mode = 'split'; // 'split', 'blend', 'checker', 'diff', 'registered'

    // Cached image elements
    this.refImg = null;
    this.warpedImg = null;
    this.diffImg = null;
    this.checkerImg = null;

    // View parameters
    this.splitPercent = 50;
    this.blendAlpha = 0.5;
    this.isDraggingSplit = false;

    this.initDOM();
  }

  initDOM() {
    this.container.innerHTML = `
      <div class="split-container" id="splitContainer">
        <div class="split-layer" id="layerUnder">
          <img id="imgUnder" alt="Reference Layer" />
        </div>
        <div class="split-before" id="layerOver">
          <img id="imgOver" alt="Registered Source Layer" />
        </div>
        <div class="split-divider-handle" id="splitHandle">↔</div>
      </div>
      <div id="singleImageViewer" style="display: none; width: 100%; height: 480px; align-items: center; justify-content: center;">
        <img id="imgSingle" style="max-width: 100%; max-height: 480px; object-fit: contain;" />
      </div>
    `;

    this.splitContainer = this.container.querySelector('#splitContainer');
    this.layerUnder = this.container.querySelector('#layerUnder');
    this.layerOver = this.container.querySelector('#layerOver');
    this.imgUnder = this.container.querySelector('#imgUnder');
    this.imgOver = this.container.querySelector('#imgOver');
    this.splitHandle = this.container.querySelector('#splitHandle');
    this.singleImageViewer = this.container.querySelector('#singleImageViewer');
    this.imgSingle = this.container.querySelector('#imgSingle');

    this.setupEvents();
  }

  setupEvents() {
    const handleMove = (clientX) => {
      const rect = this.splitContainer.getBoundingClientRect();
      let percent = ((clientX - rect.left) / rect.width) * 100;
      percent = Math.max(0, Math.min(100, percent));
      this.splitPercent = percent;
      this.updateSplitView();
    };

    this.splitContainer.addEventListener('mousedown', (e) => {
      this.isDraggingSplit = true;
      handleMove(e.clientX);
    });

    window.addEventListener('mousemove', (e) => {
      if (this.isDraggingSplit) {
        handleMove(e.clientX);
      }
    });

    window.addEventListener('mouseup', () => {
      this.isDraggingSplit = false;
    });

    // Touch support
    this.splitContainer.addEventListener('touchmove', (e) => {
      if (e.touches.length > 0) {
        handleMove(e.touches[0].clientX);
      }
    });
  }

  setImages(imageData) {
    // imageData contains { reference, warped_source, checkerboard, difference_map, alpha_blend }
    this.imageData = imageData;
    this.render();
  }

  setMode(newMode) {
    this.mode = newMode;
    this.render();
  }

  setAlpha(alphaValue) {
    this.blendAlpha = parseFloat(alphaValue);
    if (this.mode === 'blend' && this.layerOver) {
      this.layerOver.style.opacity = this.blendAlpha;
    }
  }

  updateSplitView() {
    if (this.layerOver && this.splitHandle) {
      this.layerOver.style.width = `${this.splitPercent}%`;
      this.splitHandle.style.left = `${this.splitPercent}%`;
    }
  }

  render() {
    if (!this.imageData) return;

    if (this.mode === 'split') {
      this.splitContainer.style.display = 'block';
      this.singleImageViewer.style.display = 'none';
      this.splitHandle.style.display = 'flex';

      this.imgUnder.src = this.imageData.reference;
      this.imgOver.src = this.imageData.warped_source;

      this.layerOver.style.opacity = '1.0';
      this.layerOver.style.borderRight = '2px solid var(--accent-cyan)';
      this.updateSplitView();
    } else if (this.mode === 'blend') {
      this.splitContainer.style.display = 'block';
      this.singleImageViewer.style.display = 'none';
      this.splitHandle.style.display = 'none';

      this.imgUnder.src = this.imageData.reference;
      this.imgOver.src = this.imageData.warped_source;

      this.layerOver.style.width = '100%';
      this.layerOver.style.borderRight = 'none';
      this.layerOver.style.opacity = this.blendAlpha;
    } else if (this.mode === 'checker') {
      this.splitContainer.style.display = 'none';
      this.singleImageViewer.style.display = 'flex';
      this.imgSingle.src = this.imageData.checkerboard;
    } else if (this.mode === 'diff') {
      this.splitContainer.style.display = 'none';
      this.singleImageViewer.style.display = 'flex';
      this.imgSingle.src = this.imageData.difference_map;
    } else if (this.mode === 'warped') {
      this.splitContainer.style.display = 'none';
      this.singleImageViewer.style.display = 'flex';
      this.imgSingle.src = this.imageData.warped_source;
    } else if (this.mode === 'reference') {
      this.splitContainer.style.display = 'none';
      this.singleImageViewer.style.display = 'flex';
      this.imgSingle.src = this.imageData.reference;
    }
  }
}
