interface Point {
  step: number;
  loss: number;
}

interface Props {
  data: Point[];
  height?: number;
}

function smooth(values: number[], window = 5): number[] {
  if (values.length < 2) return values;
  const out: number[] = [];
  for (let i = 0; i < values.length; i++) {
    const start = Math.max(0, i - window + 1);
    const slice = values.slice(start, i + 1);
    out.push(slice.reduce((a, b) => a + b, 0) / slice.length);
  }
  return out;
}

export default function LossChart({ data, height = 160 }: Props) {
  const width = 100;
  const pad = { t: 8, r: 4, b: 18, l: 4 };
  const innerW = width - pad.l - pad.r;
  const innerH = height - pad.t - pad.b;

  if (data.length < 2) {
    return (
      <div className="loss-chart loss-chart--empty" style={{ height }}>
        <span>Loss chart appears once training starts</span>
      </div>
    );
  }

  const losses = data.map((d) => d.loss);
  const smoothed = smooth(losses);
  const min = Math.min(...losses, ...smoothed) * 0.95;
  const max = Math.max(...losses, ...smoothed) * 1.05;
  const range = max - min || 1;

  const toX = (i: number) => pad.l + (i / (data.length - 1)) * innerW;
  const toY = (v: number) => pad.t + innerH - ((v - min) / range) * innerH;

  const rawPath = data
    .map((d, i) => `${i === 0 ? "M" : "L"} ${toX(i).toFixed(2)} ${toY(d.loss).toFixed(2)}`)
    .join(" ");

  const smoothPath = smoothed
    .map((v, i) => `${i === 0 ? "M" : "L"} ${toX(i).toFixed(2)} ${toY(v).toFixed(2)}`)
    .join(" ");

  return (
    <div className="loss-chart" style={{ height }}>
      <svg viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none">
        <defs>
          <linearGradient id="lossFill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="rgba(16, 185, 129, 0.25)" />
            <stop offset="100%" stopColor="rgba(16, 185, 129, 0)" />
          </linearGradient>
        </defs>
        {[0.25, 0.5, 0.75].map((f) => (
          <line
            key={f}
            x1={pad.l}
            x2={width - pad.r}
            y1={pad.t + innerH * f}
            y2={pad.t + innerH * f}
            className="loss-grid"
          />
        ))}
        <path d={`${smoothPath} L ${toX(data.length - 1)} ${pad.t + innerH} L ${pad.l} ${pad.t + innerH} Z`} fill="url(#lossFill)" />
        <path d={rawPath} className="loss-line loss-line--raw" vectorEffect="non-scaling-stroke" />
        <path d={smoothPath} className="loss-line loss-line--smooth" vectorEffect="non-scaling-stroke" />
      </svg>
      <div className="loss-legend">
        <span><i className="dot dot--raw" /> Loss</span>
        <span><i className="dot dot--smooth" /> Smoothed</span>
      </div>
    </div>
  );
}
