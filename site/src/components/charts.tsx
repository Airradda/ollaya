/**
 * Charts for the results page, drawn at build time as inline SVG (no client script, no chart
 * library): they read the same in dark mode through the colour tokens, and each comes with the
 * numbers as text next to or under it.
 */
import type { Child } from 'hono/jsx'

const fmt = (v: number, digits = 3) => v.toFixed(digits)

/** A number for an axis tick: 5, 20, 100, 1k, 10k. */
export function tickLabel(ms: number): string {
  if (ms >= 1000) return `${ms / 1000}k`.replace('.0k', 'k')
  return `${ms}`
}

interface Scale {
  (v: number): number
}

function logScale(min: number, max: number, from: number, to: number): Scale {
  const a = Math.log(min)
  const b = Math.log(max)
  return (v) => from + ((Math.log(Math.max(min, Math.min(max, v))) - a) / (b - a)) * (to - from)
}

function linScale(min: number, max: number, from: number, to: number): Scale {
  return (v) => from + ((v - min) / (max - min)) * (to - from)
}

/**
 * The colour classes the SVG charts use, as plain CSS: embedded in a chart published as its own
 * .svg file (the README shows them), where the site's stylesheet does not reach.
 */
const STANDALONE_CSS = `
text{font-family:ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
.fill-fg{fill:#171717}.fill-body{fill:#262626}.fill-muted{fill:#737373}.fill-canvas{fill:#fff}
.stroke-line{stroke:#e5e5e5}.stroke-line-strong{stroke:#d4d4d4}.stroke-muted{stroke:#737373}.stroke-canvas{stroke:#fff}
.bg{fill:#fff}
svg{--color-series-1:#171717;--color-series-2:#2563eb;--color-series-3:#d97706;--color-series-4:#059669;--color-series-5:#9333ea;--color-series-6:#dc2626}
@media (prefers-color-scheme:dark){
svg{--color-series-1:#fafafa;--color-series-2:#60a5fa;--color-series-3:#fbbf24;--color-series-4:#34d399;--color-series-5:#c084fc;--color-series-6:#f87171}
.fill-fg{fill:#fafafa}.fill-body{fill:#d4d4d4}.fill-muted{fill:#a3a3a3}.fill-canvas{fill:#0d1117}
.stroke-line{stroke:#262626}.stroke-line-strong{stroke:#404040}.stroke-muted{stroke:#a3a3a3}.stroke-canvas{stroke:#0d1117}
.bg{fill:#0d1117}}
`

/** Attributes and the leading children that make a chart a standalone .svg file. */
function standaloneParts(W: number, H: number, on: boolean | undefined) {
  if (!on) return { attrs: {}, head: null }
  return {
    attrs: { xmlns: 'http://www.w3.org/2000/svg', width: `${W}`, height: `${H}` },
    head: (
      <>
        <style>{STANDALONE_CSS}</style>
        <rect width={W} height={H} class="bg" />
      </>
    ),
  }
}

// --------------------------------------------------------------------------------------------
// Scatter: accuracy against latency

export interface ScatterPoint {
  label: string
  x: number
  y: number
  /** Drawn hollow and grey: a point to compare against (another server). */
  muted?: boolean
}

interface Box {
  x: number
  y: number
  w: number
  h: number
}
const overlaps = (a: Box, b: Box) => a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h

/**
 * Points with their labels, placed right, left, above or below each point, whichever first
 * clears the labels already placed and the other points.
 */
export function Scatter({
  points,
  xTicks,
  yTicks,
  xTitle,
  yTitle,
  label,
  standalone,
}: {
  points: ScatterPoint[]
  xTicks: number[]
  yTicks: number[]
  xTitle: string
  yTitle: string
  label: string
  /** Render as its own .svg file, with its colours inline. */
  standalone?: boolean
}) {
  const W = 760
  const H = 540
  const m = { l: 52, r: 16, t: 16, b: 52 }
  const x = logScale(xTicks[0]!, xTicks.at(-1)!, m.l, W - m.r)
  const y = linScale(yTicks[0]!, yTicks.at(-1)!, H - m.b, m.t)
  const charW = 6.6
  const lh = 14
  const placed: Box[] = points.map((p) => ({ x: x(p.x) - 7, y: y(p.y) - 7, w: 14, h: 14 }))
  // Candidate spots around a point, nearest first: right, left, above, below, then the diagonals,
  // at growing distances. A label further than the first ring gets a leader line.
  const dirs: [number, number][] = [
    [1, 0],
    [-1, 0],
    [0, -1],
    [0, 1],
    [1, -1],
    [1, 1],
    [-1, -1],
    [-1, 1],
  ]
  const labels = [...points]
    .sort((a, b) => b.y - a.y || a.x - b.x)
    .map((p) => {
      const px = x(p.x)
      const py = y(p.y)
      const w = p.label.length * charW
      const fits = (box: Box) =>
        box.x >= m.l + 2 && box.x + box.w <= W - 2 && box.y >= 0 && box.y + box.h <= H - m.b && !placed.some((b) => overlaps(box, b))
      for (const d of [9, 22, 36, 52, 70]) {
        for (const [dx, dy] of dirs) {
          const bx = dx > 0 ? px + d : dx < 0 ? px - d - w : px - w / 2
          const by = dy > 0 ? py + d - 3 : dy < 0 ? py - d - lh + 3 : py - lh / 2
          const box = { x: bx, y: by, w, h: lh }
          if (fits(box)) {
            placed.push(box)
            const lead = d > 9
            // The leader line ends at the label's nearest edge.
            const ex = Math.max(bx, Math.min(px, bx + w))
            const ey = Math.max(by, Math.min(py, by + lh))
            return { p, x: bx, y: by + 11, lead, ex, ey, px, py }
          }
        }
      }
      return { p, x: px + 9, y: py + 4, lead: false, ex: 0, ey: 0, px, py }
    })
  const sa = standaloneParts(W, H, standalone)
  return (
    <svg viewBox={`0 0 ${W} ${H}`} class="h-auto w-full" role="img" aria-label={label} {...sa.attrs}>
      <title>{label}</title>
      {sa.head}
      {yTicks.map((t) => (
        <g>
          <line x1={m.l} x2={W - m.r} y1={y(t)} y2={y(t)} class="stroke-line" stroke-width="1" />
          <text x={m.l - 8} y={y(t) + 4} text-anchor="end" class="fill-muted" font-size="11">
            {t.toFixed(2)}
          </text>
        </g>
      ))}
      {xTicks.map((t) => (
        <g>
          <line x1={x(t)} x2={x(t)} y1={m.t} y2={H - m.b} class="stroke-line" stroke-width="1" />
          <text x={x(t)} y={H - m.b + 18} text-anchor="middle" class="fill-muted" font-size="11">
            {tickLabel(t)}
          </text>
        </g>
      ))}
      <text x={(m.l + W - m.r) / 2} y={H - 10} text-anchor="middle" class="fill-muted" font-size="12">
        {xTitle}
      </text>
      <text x={14} y={(m.t + H - m.b) / 2} text-anchor="middle" class="fill-muted" font-size="12" transform={`rotate(-90 14 ${(m.t + H - m.b) / 2})`}>
        {yTitle}
      </text>
      {points.map((p) =>
        p.muted ? (
          <circle cx={x(p.x)} cy={y(p.y)} r="5" class="fill-canvas stroke-muted" stroke-width="2" />
        ) : (
          <circle cx={x(p.x)} cy={y(p.y)} r="5" class="fill-fg" />
        ),
      )}
      {labels.map((l) =>
        l.lead ? <line x1={l.px} y1={l.py} x2={l.ex} y2={l.ey} class="stroke-line-strong" stroke-width="1" /> : null,
      )}
      {labels.map((l) => (
        <text x={l.x} y={l.y} font-size="11" class={l.p.muted ? 'fill-muted' : 'fill-body'} font-family="ui-monospace, monospace">
          {l.p.label}
        </text>
      ))}
    </svg>
  )
}

// --------------------------------------------------------------------------------------------
// Horizontal bars with values (HTML, like the home page's scoreboard)

export interface BarRow {
  label: string
  value: number
  text: string
  muted?: boolean
  note?: string
}

export function Bars({ rows, max, label }: { rows: BarRow[]; max: number; label: string }) {
  return (
    <div class="grid grid-cols-[8.5rem_minmax(0,1fr)] gap-x-3 sm:grid-cols-[11rem_minmax(0,1fr)] sm:gap-x-4" role="list" aria-label={label}>
      {rows.map((r) => (
        <div class="contents" role="listitem">
          <span class="flex h-8 flex-col justify-center leading-tight">
            <span class={`truncate font-mono text-xs sm:text-[13px] ${r.muted ? 'text-muted' : 'text-fg'}`}>{r.label}</span>
            {r.note ? <span class="text-[11px] text-muted">{r.note}</span> : null}
          </span>
          <span class="flex h-8 items-center">
            <span
              class={`h-2.5 min-w-1 rounded-full ${r.muted ? 'bg-bar-muted' : 'bg-bar'}`}
              style={`width:${Math.max(0.5, Math.min(100, (r.value / max) * 100)).toFixed(1)}%`}
            ></span>
            <span class={`ml-2.5 text-[13px] whitespace-nowrap tabular-nums ${r.muted ? 'text-muted' : 'font-medium text-fg'}`}>{r.text}</span>
          </span>
        </div>
      ))}
    </div>
  )
}

// --------------------------------------------------------------------------------------------
// Heatmap: one row per model, one column per dataset

export function Heatmap({
  rows,
  columns,
  lo,
  hi,
  label,
}: {
  rows: { label: string; muted?: boolean; values: (number | null)[] }[]
  columns: { key: string; label: string }[]
  lo: number
  hi: number
  label: string
}) {
  const shade = (v: number) => Math.round(Math.max(0, Math.min(1, (v - lo) / (hi - lo))) * 82 + 6)
  return (
    <div class="overflow-x-auto">
      <table class="w-full border-separate border-spacing-0.5 text-center text-[11px] tabular-nums" aria-label={label}>
        <thead>
          <tr>
            <th scope="col" class="sticky left-0 z-10 bg-canvas"></th>
            {columns.map((c) => (
              <th scope="col" class="h-24 min-w-9 align-bottom font-normal text-muted">
                <span class="inline-block [writing-mode:vertical-rl] rotate-180 whitespace-nowrap">{c.label}</span>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr>
              <th scope="row" class={`sticky left-0 z-10 bg-canvas pr-2 text-left font-mono text-[11px] font-normal whitespace-nowrap sm:text-xs ${r.muted ? 'text-muted' : 'text-fg'}`}>
                {r.label}
              </th>
              {r.values.map((v) => {
                if (v === null) return <td class="rounded-sm bg-fill text-faint">·</td>
                const s = shade(v)
                return (
                  <td
                    class={`h-8 rounded-sm ${s > 50 ? 'text-canvas' : 'text-fg'}`}
                    style={`background:color-mix(in oklab, var(--color-fg) ${s}%, transparent)`}
                    title={`${r.label}: ${fmt(v)}`}
                  >
                    {Math.round(v * 100)}
                  </td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

// --------------------------------------------------------------------------------------------
// Dot plot: one row per model, one dot per device, on a log axis of milliseconds

export interface DotSeries {
  key: string
  label: string
  color: string
}

export function DotPlot({
  rows,
  series,
  ticks,
  label,
  standalone,
  legend,
}: {
  rows: { label: string; values: Record<string, number | undefined> }[]
  series: DotSeries[]
  ticks: number[]
  label: string
  /** Render as its own .svg file, with its colours inline. */
  standalone?: boolean
  /** Draw the legend inside the chart (for the standalone file). */
  legend?: boolean
}) {
  const W = 760
  const rowH = 24
  const m = { l: 150, r: 16, t: legend ? 34 : 8, b: 34 }
  const H = m.t + rows.length * rowH + m.b
  const x = logScale(ticks[0]!, ticks.at(-1)!, m.l, W - m.r)
  const sa = standaloneParts(W, H, standalone)
  let lx = m.l
  return (
    <svg viewBox={`0 0 ${W} ${H}`} class="h-auto w-full" role="img" aria-label={label} {...sa.attrs}>
      <title>{label}</title>
      {sa.head}
      {legend
        ? series.map((s) => {
            const at = lx
            lx += 34 + s.label.length * 6.6
            return (
              <g>
                <circle cx={at + 5} cy={14} r="5" style={`fill:${s.color}`} />
                <text x={at + 14} y={18} font-size="11.5" class="fill-body">
                  {s.label}
                </text>
              </g>
            )
          })
        : null}
      {ticks.map((t) => (
        <g>
          <line x1={x(t)} x2={x(t)} y1={m.t} y2={H - m.b} class="stroke-line" stroke-width="1" />
          <text x={x(t)} y={H - m.b + 16} text-anchor="middle" class="fill-muted" font-size="11">
            {tickLabel(t)}
          </text>
        </g>
      ))}
      <text x={(m.l + W - m.r) / 2} y={H - 4} text-anchor="middle" class="fill-muted" font-size="11">
        milliseconds for five questions, log scale
      </text>
      {rows.map((r, i) => {
        const cy = m.t + i * rowH + rowH / 2
        const vals = series.map((s) => r.values[s.key]).filter((v): v is number => v !== undefined)
        const lo = vals.length ? Math.min(...vals) : 0
        const hi = vals.length ? Math.max(...vals) : 0
        return (
          <g>
            <text x={m.l - 10} y={cy + 4} text-anchor="end" font-size="11.5" class="fill-fg" font-family="ui-monospace, monospace">
              {r.label}
            </text>
            {vals.length > 1 ? <line x1={x(lo)} x2={x(hi)} y1={cy} y2={cy} class="stroke-line-strong" stroke-width="2" /> : null}
            {series.map((s) => {
              const v = r.values[s.key]
              if (v === undefined) return null
              return (
                <circle cx={x(v)} cy={cy} r="5" style={`fill:${s.color}`} class="stroke-canvas" stroke-width="1.5">
                  <title>{`${r.label} on ${s.label}: ${Math.round(v)} ms`}</title>
                </circle>
              )
            })}
          </g>
        )
      })}
    </svg>
  )
}

export function Legend({ series }: { series: DotSeries[] }) {
  return (
    <ul class="flex flex-wrap gap-x-5 gap-y-2 text-[13px] text-body" role="list">
      {series.map((s) => (
        <li class="inline-flex items-center gap-2">
          <span class="inline-block size-2.5 rounded-full" style={`background:${s.color}`}></span>
          {s.label}
        </li>
      ))}
    </ul>
  )
}

/** A figure: the chart, a caption above, and optional notes under it. */
export function Figure({ title, sub, children, notes }: { title: string; sub?: string; children: Child; notes?: Child }) {
  return (
    <figure>
      <figcaption class="text-sm font-medium text-fg">
        {title}
        {sub ? <span class="font-normal text-muted"> · {sub}</span> : null}
      </figcaption>
      <div class="mt-5">{children}</div>
      {notes ? <div class="mt-4 max-w-3xl text-[13px] leading-relaxed text-muted">{notes}</div> : null}
    </figure>
  )
}
