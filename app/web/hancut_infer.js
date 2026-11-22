// 한컷점검 브라우저 판정 (#62) — E1 ONNX 를 onnxruntime-web(wasm)으로 돌린다. 사진은 밖으로 나가지 않는다.
// 서버 `server/app/dense.py` 와 같은 규칙: cv2 디코드(BGR→RGB) → Ultralytics LetterBox(640, INTER_LINEAR, 114)
// → /255 → 출력 1×16×8400 에서 품목마다 앵커 최댓값 = 점수, 그 앵커의 박스 = 가장 의심되는 자리.
// 점수는 .9g 로 자르고 dense bundle 의 품목별 low/high 로 통과 · 재검 · 적발을 정한다.
(function () {
  const ORT_VERSION = '1.20.1';
  const ORT_CDN = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
  const SIZE = 640, ANCHORS = 8400, PAD = 114;
  let ready = null, session = null, bundle = null;

  function loadScript(src) {
    return new Promise((resolve, reject) => {
      const s = document.createElement('script');
      s.src = src; s.onload = resolve; s.onerror = () => reject(new Error('판정 엔진을 불러오지 못했어요.'));
      document.head.append(s);
    });
  }

  function warmup() {
    if (!ready) {
      ready = (async () => {
        if (!window.ort) await loadScript(ORT_CDN + 'ort.min.js');
        ort.env.wasm.wasmPaths = ORT_CDN;
        const res = await fetch('models/bundle.json');
        if (!res.ok) throw new Error('판정 기준(bundle)을 불러오지 못했어요.');
        bundle = await res.json();
        session = await ort.InferenceSession.create('models/e1.onnx', { executionProviders: ['wasm'] });
      })();
      ready.catch(() => { ready = null; });
    }
    return ready;
  }

  // 파이썬 round 와 같은 짝수 반올림 (LetterBox 크기 · 여백 계산)
  function roundHalfEven(x) {
    const f = Math.floor(x), d = x - f;
    if (Math.abs(d - 0.5) < 1e-9) return f % 2 === 0 ? f : f + 1;
    return Math.round(x);
  }

  // cv2.resize INTER_LINEAR — OpenCV 5.1 은 가로로 먼저 보간해 8비트로 반올림한 뒤 세로로 보간한다.
  // 축마다 11비트 고정소수 계수. 서버와 픽셀 0.3% 정도만 ±1 차이 난다 (둘을 한 번에 하면 5%).
  function taps(dn, sn) {
    const o = new Int32Array(dn), b = new Int32Array(dn), sc = sn / dn;
    for (let x = 0; x < dn; x++) {
      let f = (x + 0.5) * sc - 0.5, s = Math.floor(f); f -= s;
      if (s < 0) { s = 0; f = 0; }
      if (s >= sn - 1) { s = sn - 1; f = 0; }
      o[x] = s; b[x] = Math.round(f * 2048);
    }
    return { o, b };
  }

  function resizeLinear(src, sw, sh, dw, dh) {
    const tx = taps(dw, sw), ty = taps(dh, sh);
    const mid = new Uint8ClampedArray(sh * dw * 3);
    for (let y = 0; y < sh; y++) for (let x = 0; x < dw; x++) {
      const s = tx.o[x], s1 = Math.min(s + 1, sw - 1), b = tx.b[x], a = 2048 - b;
      const p0 = (y * sw + s) * 4, p1 = (y * sw + s1) * 4, d = (y * dw + x) * 3;
      for (let c = 0; c < 3; c++) mid[d + c] = (src[p0 + c] * a + src[p1 + c] * b + 1024) >> 11;
    }
    const out = new Uint8ClampedArray(dw * dh * 3), row = dw * 3;
    for (let y = 0; y < dh; y++) {
      const s = ty.o[y], s1 = Math.min(s + 1, sh - 1), b = ty.b[y], a = 2048 - b;
      for (let i = 0; i < row; i++) out[y * row + i] = (mid[s * row + i] * a + mid[s1 * row + i] * b + 1024) >> 11;
    }
    return out;
  }

  async function decode(bytes) {
    const bmp = await createImageBitmap(new Blob([bytes]), { colorSpaceConversion: 'none', premultiplyAlpha: 'none' });
    const canvas = document.createElement('canvas');
    canvas.width = bmp.width; canvas.height = bmp.height;
    const ctx = canvas.getContext('2d', { willReadFrequently: true, colorSpace: 'srgb' });
    ctx.drawImage(bmp, 0, 0);
    const { data } = ctx.getImageData(0, 0, bmp.width, bmp.height);
    return { data, width: bmp.width, height: bmp.height };
  }

  function letterbox(img) {
    const r = Math.min(SIZE / img.height, SIZE / img.width);
    const nw = roundHalfEven(img.width * r), nh = roundHalfEven(img.height * r);
    const left = roundHalfEven((SIZE - nw) / 2 - 0.1), top = roundHalfEven((SIZE - nh) / 2 - 0.1);
    let rgb;
    if (nw === img.width && nh === img.height) {
      rgb = new Uint8ClampedArray(nw * nh * 3);
      for (let i = 0; i < nw * nh; i++) for (let c = 0; c < 3; c++) rgb[i * 3 + c] = img.data[i * 4 + c];
    } else {
      rgb = resizeLinear(img.data, img.width, img.height, nw, nh);
    }
    const x = new Float32Array(3 * SIZE * SIZE).fill(PAD / 255);
    for (let y = 0; y < nh; y++) for (let xx = 0; xx < nw; xx++) {
      const s = (y * nw + xx) * 3, d = (y + top) * SIZE + (xx + left);
      x[d] = rgb[s] / 255; x[SIZE * SIZE + d] = rgb[s + 1] / 255; x[2 * SIZE * SIZE + d] = rgb[s + 2] / 255;
    }
    return { tensor: x, r, left, top };
  }

  async function inspect(bytes) {
    await warmup();
    const img = await decode(bytes);
    const { tensor, r, left, top } = letterbox(img);
    const out = await session.run({ [session.inputNames[0]]: new ort.Tensor('float32', tensor, [1, 3, SIZE, SIZE]) });
    const y = out[session.outputNames[0]].data;
    const names = bundle.class_names;
    if (y.length !== (4 + names.length) * ANCHORS) throw new Error('지원하지 않는 모델 출력이에요.');
    const clamp = (v, hi) => Math.round(Math.min(Math.max(v, 0), hi) * 10) / 10;
    const items = names.map((name, c) => {
      let best = -1, at = 0;
      for (let k = 0, o = (4 + c) * ANCHORS; k < ANCHORS; k++) if (y[o + k] > best) { best = y[o + k]; at = k; }
      if (!(best >= 0 && best <= 1)) throw new Error('잘못된 점수가 나왔어요.');
      const score = Number(best.toPrecision(9));
      const t = bundle.thresholds[name] || bundle.thresholds.default;
      const zone = score < t.low ? 'auto_clear' : score >= t.high ? 'auto_alarm' : 'review';
      const [cx, cy, w, h] = [0, 1, 2, 3].map(i => y[i * ANCHORS + at]);
      const box = [clamp((cx - w / 2 - left) / r, img.width), clamp((cy - h / 2 - top) / r, img.height),
                   clamp((cx + w / 2 - left) / r, img.width), clamp((cy + h / 2 - top) / r, img.height)];
      return { item: name, score, zone, needs_review: zone === 'review', thresholds: { low: t.low, high: t.high }, box };
    });
    return JSON.stringify({ items, image_size: [img.width, img.height], model_version: `sha256:${bundle.weights_sha256} · 브라우저 ONNX`,
                            experimental: true, engine: 'browser' });
  }

  window.hancutWarmup = () => warmup().then(() => 'ok');
  window.hancutInspect = (bytes) => inspect(bytes);
})();
