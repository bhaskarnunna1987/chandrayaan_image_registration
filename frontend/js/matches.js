/**
 * matches.js
 * Canvas rendering engine for correspondence lines, 8x8 spatial distribution grid,
 * and sub-pixel displacement quiver vectors.
 */

class MatchesCanvasRenderer {
  constructor(canvasElement) {
    this.canvas = canvasElement;
    this.ctx = canvasElement.getContext('2d');
    this.imgSrc = null;
    this.imgRef = null;
    this.pointsData = null;
    this.spatialData = null;

    // View toggles
    this.showInliersOnly = true;
    this.showSpatialGrid = false;
    this.showSubpixelVectors = false;
    this.vectorScale = 8.0; // Magnification factor for sub-pixel quiver arrows
  }

  setImages(srcImgElement, refImgElement) {
    this.imgSrc = srcImgElement;
    this.imgRef = refImgElement;
  }

  setData(pointsData, spatialData) {
    this.pointsData = pointsData;
    this.spatialData = spatialData;
  }

  render() {
    if (!this.imgSrc || !this.imgRef || !this.pointsData) return;

    const ctx = this.ctx;
    const canvas = this.canvas;

    // Target dimensions: side by side
    const maxH = 500;
    const aspectSrc = this.imgSrc.naturalWidth / this.imgSrc.naturalHeight;
    const aspectRef = this.imgRef.naturalWidth / this.imgRef.naturalHeight;

    const h = Math.min(maxH, Math.max(this.imgSrc.naturalHeight, this.imgRef.naturalHeight));
    const wSrc = h * aspectSrc;
    const wRef = h * aspectRef;
    const gap = 30;

    canvas.width = wSrc + wRef + gap;
    canvas.height = h;

    // Clear canvas
    ctx.fillStyle = '#04070d';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Draw Source Image (Left)
    ctx.drawImage(this.imgSrc, 0, 0, wSrc, h);

    // Draw Divider & Reference Image (Right)
    const refXOffset = wSrc + gap;
    ctx.fillStyle = '#0e1424';
    ctx.fillRect(wSrc, 0, gap, h);

    ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(wSrc + gap / 2, 0);
    ctx.lineTo(wSrc + gap / 2, h);
    ctx.stroke();

    ctx.drawImage(this.imgRef, refXOffset, 0, wRef, h);

    // Scale factors from natural image coordinates to canvas coordinates
    const scaleSrcX = wSrc / this.imgSrc.naturalWidth;
    const scaleSrcY = h / this.imgSrc.naturalHeight;

    const scaleRefX = wRef / this.imgRef.naturalWidth;
    const scaleRefY = h / this.imgRef.naturalHeight;

    // Optional: Draw 8x8 Spatial Grid on Reference Image
    if (this.showSpatialGrid && this.spatialData) {
      this.drawSpatialGrid(ctx, refXOffset, 0, wRef, h);
    }

    // Draw Correspondence Lines & Keypoint Markers
    const selSrc = this.pointsData.selected_src || [];
    const selRef = this.pointsData.selected_ref || [];
    const refRef = this.pointsData.refined_ref || [];
    const disps = this.pointsData.displacements || [];

    const numPoints = Math.min(selSrc.length, selRef.length);

    for (let i = 0; i < numPoints; i++) {
      const pSrc = selSrc[i];
      const pRef = selRef[i];

      const x1 = pSrc[0] * scaleSrcX;
      const y1 = pSrc[1] * scaleSrcY;

      const x2 = refXOffset + pRef[0] * scaleRefX;
      const y2 = pRef[1] * scaleRefY;

      // Correspondence connecting line with subtle gradient
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.lineWidth = 1.2;
      ctx.strokeStyle = 'rgba(0, 229, 255, 0.45)';
      ctx.stroke();

      // Source point dot (Emerald green)
      ctx.beginPath();
      ctx.arc(x1, y1, 3.5, 0, 2 * Math.PI);
      ctx.fillStyle = '#10b981';
      ctx.fill();
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 1;
      ctx.stroke();

      // Reference point dot (Cyan)
      ctx.beginPath();
      ctx.arc(x2, y2, 3.5, 0, 2 * Math.PI);
      ctx.fillStyle = '#00e5ff';
      ctx.fill();
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 1;
      ctx.stroke();

      // Sub-pixel Quiver Vector overlay
      if (this.showSubpixelVectors && disps[i]) {
        const dx = disps[i][0] * this.vectorScale * scaleRefX;
        const dy = disps[i][1] * this.vectorScale * scaleRefY;

        if (Math.abs(dx) > 0.05 || Math.abs(dy) > 0.05) {
          // Draw magnified shift arrow in bright yellow/amber
          ctx.beginPath();
          ctx.moveTo(x2, y2);
          ctx.lineTo(x2 + dx, y2 + dy);
          ctx.strokeStyle = '#fbbf24';
          ctx.lineWidth = 2.0;
          ctx.stroke();

          // Refined sub-pixel target dot
          ctx.beginPath();
          ctx.arc(x2 + dx, y2 + dy, 2.5, 0, 2 * Math.PI);
          ctx.fillStyle = '#f59e0b';
          ctx.fill();
        }
      }
    }

    // Overlay Label Tags
    ctx.fillStyle = 'rgba(7, 10, 18, 0.75)';
    ctx.fillRect(10, 10, 140, 26);
    ctx.fillRect(refXOffset + 10, 10, 160, 26);

    ctx.fillStyle = '#f8fafc';
    ctx.font = '600 12px Inter, sans-serif';
    ctx.fillText('SOURCE (OHRC / Moving)', 16, 27);
    ctx.fillText('REFERENCE (TMC / Fixed)', refXOffset + 16, 27);
  }

  drawSpatialGrid(ctx, ox, oy, w, h) {
    const rows = 8;
    const cols = 8;
    const cellW = w / cols;
    const cellH = h / rows;

    const counts = this.spatialData.grid_cell_counts || [];

    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const x = ox + c * cellW;
        const y = oy + r * cellH;
        const count = (counts[r] && counts[r][c]) ? counts[r][c] : 0;

        // Color cell background if occupied
        if (count > 0) {
          ctx.fillStyle = 'rgba(0, 229, 255, 0.12)';
          ctx.fillRect(x, y, cellW, cellH);

          // Draw small count tag in top-right of cell
          ctx.fillStyle = '#00e5ff';
          ctx.font = '10px monospace';
          ctx.fillText(count.toString(), x + 4, y + 12);
        }

        // Cell boundary
        ctx.strokeStyle = count > 0 ? 'rgba(0, 229, 255, 0.35)' : 'rgba(255, 255, 255, 0.06)';
        ctx.lineWidth = 1;
        ctx.strokeRect(x, y, cellW, cellH);
      }
    }
  }
}
