/**
 * app.js
 * Mission Control Application Controller for Chandrayaan-2 Lunar Image Registration.
 * Handles state, REST APIs, telemetry dashboards, KPI updates, and exports.
 */

document.addEventListener('DOMContentLoaded', () => {
  // Global Application State
  const state = {
    demoCaseId: 'sun_angle_variation',
    sourceUploadedFilename: null,
    referenceUploadedFilename: null,
    activeTab: 'matches', // 'matches' or 'registration'
    latestResult: null
  };

  // DOM Elements
  const demoCaseSelect = document.getElementById('demoCaseSelect');
  const demoCaseDesc = document.getElementById('demoCaseDesc');
  const btnRunPipeline = document.getElementById('btnRunPipeline');
  const loadingOverlay = document.getElementById('loadingOverlay');
  const loadingStatusText = document.getElementById('loadingStatusText');
  const watermarkBanner = document.getElementById('watermarkBanner');

  // Algorithm parameter inputs
  const selectDetector = document.getElementById('selectDetector');
  const selectSensorSrc = document.getElementById('selectSensorSrc');
  const selectSensorRef = document.getElementById('selectSensorRef');
  const selectModel = document.getElementById('selectModel');
  const toggleHomomorphic = document.getElementById('toggleHomomorphic');
  const toggleStructural = document.getElementById('toggleStructural');
  const toggleMultiscale = document.getElementById('toggleMultiscale');
  const inputRatioThresh = document.getElementById('inputRatioThresh');
  const ratioThreshVal = document.getElementById('ratioThreshVal');
  const inputGridSize = document.getElementById('inputGridSize');
  const gridSizeVal = document.getElementById('gridSizeVal');

  // Visualizer Toggles
  const toggleSpatialGrid = document.getElementById('toggleSpatialGrid');
  const toggleSubpixelVectors = document.getElementById('toggleSubpixelVectors');
  const blendOpacitySlider = document.getElementById('blendOpacitySlider');
  const blendOpacityVal = document.getElementById('blendOpacityVal');

  // KPI Elements
  const kpiInlierRatio = document.getElementById('kpiInlierRatio');
  const kpiRmse = document.getElementById('kpiRmse');
  const kpiCoverage = document.getElementById('kpiCoverage');
  const kpiLatency = document.getElementById('kpiLatency');
  const kpiInliersCount = document.getElementById('kpiInliersCount');
  const kpiSubpixelShift = document.getElementById('kpiSubpixelShift');

  // Viewport Tab Buttons
  const tabMatches = document.getElementById('tabMatches');
  const tabRegistration = document.getElementById('tabRegistration');
  const matchesSection = document.getElementById('matchesSection');
  const registrationSection = document.getElementById('registrationSection');

  // Registration Mode Buttons
  const btnModeSplit = document.getElementById('btnModeSplit');
  const btnModeBlend = document.getElementById('btnModeBlend');
  const btnModeChecker = document.getElementById('btnModeChecker');
  const btnModeDiff = document.getElementById('btnModeDiff');
  const btnModeWarped = document.getElementById('btnModeWarped');

  // Export Buttons
  const btnExportCsv = document.getElementById('btnExportCsv');
  const btnExportJson = document.getElementById('btnExportJson');
  const btnExportImage = document.getElementById('btnExportImage');

  // Previews
  const srcThumb = document.getElementById('srcThumb');
  const refThumb = document.getElementById('refThumb');
  const fileInputSrc = document.getElementById('fileInputSrc');
  const fileInputRef = document.getElementById('fileInputRef');
  const slotSrc = document.getElementById('slotSrc');
  const slotRef = document.getElementById('slotRef');

  // Instantiate Renderers
  const matchesCanvas = document.getElementById('matchesCanvas');
  const matchesRenderer = new MatchesCanvasRenderer(matchesCanvas);

  const regViewerContainer = document.getElementById('regViewerContainer');
  const regViewer = new RegistrationViewer(regViewerContainer);

  // Hidden image buffers for canvas drawing
  const imgBufferSrc = new Image();
  const imgBufferRef = new Image();
  matchesRenderer.setImages(imgBufferSrc, imgBufferRef);

  // Initial Setup: Fetch Demo Cases
  async function initDemoCases() {
    try {
      const res = await fetch('/api/demo/cases');
      const data = await res.json();
      demoCaseSelect.innerHTML = '';

      data.cases.forEach((c) => {
        const opt = document.createElement('option');
        opt.value = c.id;
        opt.textContent = c.title;
        demoCaseSelect.appendChild(opt);
      });

      // Default load first case
      if (data.cases.length > 0) {
        state.demoCaseId = data.cases[0].id;
        demoCaseDesc.textContent = data.cases[0].challenge;
        await loadDemoCase(state.demoCaseId);
      }
    } catch (err) {
      console.error('Failed to load demo cases:', err);
    }
  }

  // Load a demo case
  async function loadDemoCase(caseId) {
    showLoading('Loading synthetic benchmark dataset...');
    try {
      const res = await fetch(`/api/demo/load/${caseId}`);
      const data = await res.json();

      srcThumb.src = data.source_preview;
      srcThumb.style.display = 'block';
      refThumb.src = data.reference_preview;
      refThumb.style.display = 'block';

      imgBufferSrc.src = data.source_preview;
      imgBufferRef.src = data.reference_preview;

      state.sourceUploadedFilename = null;
      state.referenceUploadedFilename = null;

      watermarkBanner.style.display = 'flex';

      // Automatically execute pipeline on demo load for instantaneous preview
      await executePipeline();
    } catch (err) {
      console.error('Error loading demo case:', err);
    } finally {
      hideLoading();
    }
  }

  // Execute Pipeline
  async function executePipeline() {
    showLoading('Executing multi-modal correspondence pipeline...');
    try {
      const payload = {
        demo_case_id: (state.sourceUploadedFilename && state.referenceUploadedFilename) ? null : state.demoCaseId,
        source_filename: state.sourceUploadedFilename,
        reference_filename: state.referenceUploadedFilename,
        detector_type: selectDetector.value,
        sensor_src: selectSensorSrc.value,
        sensor_ref: selectSensorRef.value,
        model_type: selectModel.value,
        use_homomorphic: toggleHomomorphic.checked,
        use_structural: toggleStructural.checked,
        use_multiscale: toggleMultiscale.checked,
        ratio_threshold: parseFloat(inputRatioThresh.value),
        ransac_thresh: 3.0,
        grid_size: parseInt(inputGridSize.value)
      };

      const res = await fetch('/api/pipeline/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      const data = await res.json();

      if (!data.success) {
        alert('Pipeline Verification Notice: ' + (data.error_message || 'Could not find reliable correspondences.'));
        hideLoading();
        return;
      }

      state.latestResult = data;

      // Update Watermark Banner
      watermarkBanner.style.display = data.is_demonstration_data ? 'flex' : 'none';

      // Update KPI Cards
      const m = data.metrics.matching;
      const s = data.metrics.spatial;
      const sub = data.metrics.subpixel;
      const sim = data.metrics.image_similarity;
      const t = data.timings;

      kpiInlierRatio.textContent = `${m.inlier_ratio_pct}%`;
      kpiRmse.textContent = `${m.inlier_rmse_px} px`;
      kpiCoverage.textContent = `${s.spatial_coverage_pct}%`;
      kpiLatency.textContent = `${(t.total_pipeline_ms / 1000).toFixed(2)}s`;
      kpiInliersCount.textContent = `${m.inlier_count} / ${data.telemetry.candidate_matches}`;
      kpiSubpixelShift.textContent = `${sub.mean_subpixel_shift_px} px (${sub.refinement_rate_pct}%)`;

      // Update Detailed Metrics Table
      document.getElementById('metricDetector').textContent = selectDetector.value;
      document.getElementById('metricModel').textContent = m.model_type;
      document.getElementById('metricMeanError').textContent = `${m.mean_reproj_error_px} px`;
      document.getElementById('metricMedianError').textContent = `${m.median_reproj_error_px} px`;
      document.getElementById('metricEntropy').textContent = `${s.spatial_entropy} bits (${s.normalized_entropy})`;
      document.getElementById('metricOccupiedCells').textContent = `${s.occupied_cells} / ${s.total_cells}`;
      document.getElementById('metricNccGain').textContent = `${sim.pre_registration.ncc} → ${sim.post_registration.ncc} (Δ ${sim.ncc_gain > 0 ? '+' : ''}${sim.ncc_gain})`;
      document.getElementById('metricSsimGain').textContent = `${sim.pre_registration.ssim} → ${sim.post_registration.ssim} (Δ ${sim.ssim_gain > 0 ? '+' : ''}${sim.ssim_gain})`;

      // Update Timing Breakdown Bar
      document.getElementById('timePreprocess').textContent = `${t.preprocessing || 0} ms`;
      document.getElementById('timeFeatures').textContent = `${t.feature_extraction || 0} ms`;
      document.getElementById('timeMatching').textContent = `${t.matching || 0} ms`;
      document.getElementById('timeRansac').textContent = `${t.geometric_verification || 0} ms`;
      document.getElementById('timeSpatial').textContent = `${t.spatial_selection || 0} ms`;
      document.getElementById('timeSubpixel').textContent = `${t.subpixel_refinement || 0} ms`;
      document.getElementById('timeWarp').textContent = `${t.registration_warping || 0} ms`;

      // Update Canvas Viewers
      imgBufferSrc.onload = () => {
        matchesRenderer.setData(data.points, data.metrics.spatial);
        matchesRenderer.render();
      };
      imgBufferSrc.src = data.images.source;
      imgBufferRef.src = data.images.reference;

      // Update Registration View
      regViewer.setImages(data.images);

    } catch (err) {
      console.error('Error running pipeline:', err);
      alert('Network or execution error while running registration pipeline.');
    } finally {
      hideLoading();
    }
  }

  // Handle Image Uploads
  async function handleUpload(file, role) {
    if (!file) return;
    const formData = new FormData();
    formData.append('image', file);
    formData.append('role', role);

    showLoading(`Uploading and parsing ${role} imagery...`);
    try {
      const res = await fetch('/api/upload', {
        method: 'POST',
        body: formData
      });
      const data = await res.json();
      if (res.ok && data.status === 'success') {
        if (role === 'source') {
          state.sourceUploadedFilename = data.filename;
          srcThumb.src = data.preview;
          srcThumb.style.display = 'block';
          slotSrc.style.border = '2px solid var(--accent-cyan)';
          slotSrc.style.background = 'rgba(0, 229, 255, 0.08)';
        } else {
          state.referenceUploadedFilename = data.filename;
          refThumb.src = data.preview;
          refThumb.style.display = 'block';
          slotRef.style.border = '2px solid var(--accent-isro-orange)';
          slotRef.style.background = 'rgba(255, 123, 0, 0.08)';
        }

        if (state.sourceUploadedFilename && state.referenceUploadedFilename) {
          watermarkBanner.style.display = 'none';
        }

        // Auto-run pipeline with newly uploaded imagery
        await executePipeline();
      } else {
        alert(data.error || 'Upload failed');
      }
    } catch (err) {
      console.error('Upload failed:', err);
    } finally {
      hideLoading();
    }
  }

  // Event Listeners
  demoCaseSelect.addEventListener('change', (e) => {
    state.demoCaseId = e.target.value;
    state.sourceUploadedFilename = null;
    state.referenceUploadedFilename = null;
    slotSrc.style.border = '';
    slotSrc.style.background = '';
    slotRef.style.border = '';
    slotRef.style.background = '';
    loadDemoCase(state.demoCaseId);
  });

  const btnResetImages = document.getElementById('btnResetImages');
  if (btnResetImages) {
    btnResetImages.addEventListener('click', () => {
      state.sourceUploadedFilename = null;
      state.referenceUploadedFilename = null;
      fileInputSrc.value = '';
      fileInputRef.value = '';
      slotSrc.style.border = '';
      slotSrc.style.background = '';
      slotRef.style.border = '';
      slotRef.style.background = '';
      loadDemoCase(state.demoCaseId);
    });
  }

  btnRunPipeline.addEventListener('click', () => {
    executePipeline();
  });

  slotSrc.addEventListener('click', () => fileInputSrc.click());
  slotRef.addEventListener('click', () => fileInputRef.click());

  fileInputSrc.addEventListener('change', (e) => {
    if (e.target.files.length > 0) handleUpload(e.target.files[0], 'source');
  });

  fileInputRef.addEventListener('change', (e) => {
    if (e.target.files.length > 0) handleUpload(e.target.files[0], 'reference');
  });

  // Slider controls
  inputRatioThresh.addEventListener('input', (e) => {
    ratioThreshVal.textContent = e.target.value;
  });

  inputGridSize.addEventListener('input', (e) => {
    gridSizeVal.textContent = `${e.target.value}x${e.target.value}`;
  });

  // Toggle buttons
  toggleSpatialGrid.addEventListener('change', (e) => {
    matchesRenderer.showSpatialGrid = e.target.checked;
    matchesRenderer.render();
  });

  toggleSubpixelVectors.addEventListener('change', (e) => {
    matchesRenderer.showSubpixelVectors = e.target.checked;
    matchesRenderer.render();
  });

  blendOpacitySlider.addEventListener('input', (e) => {
    blendOpacityVal.textContent = Math.round(e.target.value * 100) + '%';
    regViewer.setAlpha(e.target.value);
  });

  // Viewport Tab Switch
  tabMatches.addEventListener('click', () => {
    tabMatches.classList.add('active');
    tabRegistration.classList.remove('active');
    matchesSection.style.display = 'flex';
    registrationSection.style.display = 'none';
    matchesRenderer.render();
  });

  tabRegistration.addEventListener('click', () => {
    tabRegistration.classList.add('active');
    tabMatches.classList.remove('active');
    matchesSection.style.display = 'none';
    registrationSection.style.display = 'flex';
    regViewer.render();
  });

  // Registration Mode Switches
  const regModeButtons = [
    { btn: btnModeSplit, mode: 'split' },
    { btn: btnModeBlend, mode: 'blend' },
    { btn: btnModeChecker, mode: 'checker' },
    { btn: btnModeDiff, mode: 'diff' },
    { btn: btnModeWarped, mode: 'warped' }
  ];

  regModeButtons.forEach(({ btn, mode }) => {
    btn.addEventListener('click', () => {
      regModeButtons.forEach(b => b.btn.classList.remove('active'));
      btn.classList.add('active');
      regViewer.setMode(mode);

      // Show opacity slider only in blend mode
      const blendControl = document.getElementById('blendOpacityControl');
      blendControl.style.display = mode === 'blend' ? 'flex' : 'none';
    });
  });

  // Exports
  btnExportCsv.addEventListener('click', () => {
    window.location.href = '/api/export/tiepoints?format=csv';
  });

  btnExportJson.addEventListener('click', () => {
    window.location.href = '/api/export/tiepoints?format=json';
  });

  btnExportImage.addEventListener('click', () => {
    window.location.href = '/api/export/registered';
  });

  function showLoading(msg) {
    loadingStatusText.textContent = msg;
    loadingOverlay.style.display = 'flex';
  }

  function hideLoading() {
    loadingOverlay.style.display = 'none';
  }

  // Kick off application
  initDemoCases();
});
