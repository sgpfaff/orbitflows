"""Build the theory note: note_data.json (make_note_data.py) + note.html + the three diagrams drawn here
-> why_energy_suffices.html."""
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def arrow(x, y, dx, dy, cls, L=30):
    n = math.hypot(dx, dy); dx, dy = dx / n, dy / n
    x0, y0, x1, y1 = x - 0.5 * L * dx, y - 0.5 * L * dy, x + 0.5 * L * dx, y + 0.5 * L * dy
    return f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y1:.1f}" class="{cls}" marker-end="url(#ah-{cls})"/>'


def defs(*classes):
    m = "".join(f'<marker id="ah-{c}" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
                f'<path d="M0 0L10 5L0 10z" class="{c}-head"/></marker>' for c in classes)
    return f"<defs>{m}</defs>"


def fig_phase():
    """One degree of freedom: a closed curve that crosses energy contours against one that is a contour."""
    W, H, b = 720, 300, 0.5
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Left: a closed curve crossing the level curves of H, with the flow crossing it. Right: a closed curve that is a level curve of H, with the flow running along it.">', defs("d-flow")]
    for cx, mode in ((180, "cross"), (540, "level")):
        cy = 158
        for a in (40, 70, 100, 130, 160):
            out.append(f'<ellipse cx="{cx}" cy="{cy}" rx="{a}" ry="{a * b:.1f}" class="d-contour"/>')
        pts = []
        if mode == "cross":
            ccx, r = cx + 34, 60
            out.append(f'<circle cx="{ccx}" cy="{cy}" r="{r}" class="d-curve"/>')
            for t in range(0, 360, 45):
                pts.append((ccx + r * math.cos(math.radians(t + 20)), cy + r * math.sin(math.radians(t + 20))))
        else:
            a = 100
            out.append(f'<ellipse cx="{cx}" cy="{cy}" rx="{a}" ry="{a * b:.1f}" class="d-curve"/>')
            for t in range(0, 360, 45):
                pts.append((cx + a * math.cos(math.radians(t + 20)), cy + a * b * math.sin(math.radians(t + 20))))
        for x, y in pts:  # the flow is tangent to the level curve through the point, clockwise on screen
            gx, gy = (x - cx), (y - cy) / b ** 2
            tx, ty = -gy, gx
            if (x - cx) * ty - (y - cy) * tx < 0:
                tx, ty = -tx, -ty
            out.append(arrow(x, y, tx, ty, "d-flow"))
        lab = "H varies along the curve: the flow crosses it" if mode == "cross" else "H constant on the curve: the flow runs along it"
        out.append(f'<text x="{cx}" y="26" text-anchor="middle" class="d-title">{lab}</text>')
        out.append(f'<text x="{cx}" y="{H - 8}" text-anchor="middle" class="d-note">position q →</text>')
        out.append(f'<text x="{cx - 170}" y="{cy}" text-anchor="middle" class="d-note" transform="rotate(-90 {cx - 170} {cy})">momentum p →</text>')
    out.append("</svg>")
    return "".join(out)


def fig_action():
    """Action space: the deformed torus is a segment along the integer vector k; energy contours are perpendicular to Omega."""
    W, H = 720, 320
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Two panels in action space. Left: a torus smeared along k = (1, 0) crosses the energy contours. Right: smeared along k = (1, -2), which is perpendicular to the frequency vector, it lies along a contour.">',
           defs("d-ink", "d-om")]
    om = (1.0, 0.5); n = math.hypot(*om); om = (om[0] / n, om[1] / n)
    for ox, k, lab, sub in ((40, (1, 0), "k = (1, 0): seen", "energy changes by ε k·Ω"), (400, (1, -2), "k = (1, −2): not seen", "k·Ω = 0, energy unchanged at first order")):
        x0, y0, w, h = ox + 30, 50, 270, 220          # plot box; I1 to the right, I2 up
        cx, cy = x0 + w / 2, y0 + h / 2
        out.append(f'<line x1="{x0}" y1="{y0 + h}" x2="{x0 + w}" y2="{y0 + h}" class="d-ink" marker-end="url(#ah-d-ink)"/>')
        out.append(f'<line x1="{x0}" y1="{y0 + h}" x2="{x0}" y2="{y0}" class="d-ink" marker-end="url(#ah-d-ink)"/>')
        out.append(f'<text x="{x0 + w}" y="{y0 + h + 18}" text-anchor="end" class="d-note">I₁ (radial action)</text>')
        out.append(f'<text x="{x0 - 10}" y="{y0 + 4}" text-anchor="end" class="d-note" transform="rotate(-90 {x0 - 10} {y0 + 4})">I₂ (angular momentum)</text>')
        tx, ty = -om[1], om[0]                         # along a contour (math coords); screen y is flipped below
        for s in (-90, -45, 0, 45, 90):                # contours of h, spaced along Omega
            px, py = cx + s * om[0], cy - s * om[1]
            a = 150
            xa, ya, xb, yb = px - a * tx, py + a * ty, px + a * tx, py - a * ty
            out.append(f'<line x1="{xa:.1f}" y1="{ya:.1f}" x2="{xb:.1f}" y2="{yb:.1f}" class="d-contour" clip-path="url(#clip{ox})"/>')
        out.append(f'<clipPath id="clip{ox}"><rect x="{x0 + 2}" y="{y0}" width="{w - 4}" height="{h - 2}"/></clipPath>')
        out.append(f'<line x1="{cx}" y1="{cy}" x2="{cx + 74 * om[0]:.1f}" y2="{cy - 74 * om[1]:.1f}" class="d-om" marker-end="url(#ah-d-om)"/>')
        out.append(f'<text x="{cx + 82 * om[0]:.1f}" y="{cy - 82 * om[1] + 4:.1f}" class="d-om-text">Ω</text>')
        kn = math.hypot(*k); L = 58
        dx, dy = L * k[0] / kn, -L * k[1] / kn
        out.append(f'<line x1="{cx - dx:.1f}" y1="{cy - dy:.1f}" x2="{cx + dx:.1f}" y2="{cy + dy:.1f}" class="d-torus"/>')
        out.append(f'<circle cx="{cx}" cy="{cy}" r="4" class="d-dot"/>')
        out.append(f'<text x="{cx - 8}" y="{cy + 18}" text-anchor="end" class="d-note">I₀</text>')
        out.append(f'<text x="{x0 + w / 2}" y="24" text-anchor="middle" class="d-title">{lab}</text>')
        out.append(f'<text x="{x0 + w / 2}" y="40" text-anchor="middle" class="d-note">{sub}</text>')
    out.append(f'<text x="{40 + 30 + 262}" y="{50 + 28}" text-anchor="end" class="d-note">contours of h(I)</text>')
    out.append("</svg>")
    return "".join(out)


def fig_sphere(tilt=11.3):
    """Directions of L round the third angle: a circle about z (conventional) and about a tilted axis (learned)."""
    W, H, cx, cy, r, inc = 720, 300, 250, 158, 112, 50.0
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="A sphere of angular-momentum directions. The conventional torus traces a circle of constant inclination about the z axis; the learned torus traces the same circle about an axis tilted by {tilt:.0f} degrees.">',
           defs("d-ink", "d-om")]
    out.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" class="d-contour"/>')
    ey, rx = r * math.cos(math.radians(inc)), r * math.sin(math.radians(inc))

    def ring(cls, rot):
        g = f'<g transform="rotate({rot} {cx} {cy})">'
        g += f'<ellipse cx="{cx}" cy="{cy - ey:.1f}" rx="{rx:.1f}" ry="{rx * 0.26:.1f}" class="{cls}"/>'
        g += f'<line x1="{cx}" y1="{cy + r + 14}" x2="{cx}" y2="{cy - r - 22}" class="{"d-ink" if cls == "d-ring" else "d-om"}" marker-end="url(#ah-{"d-ink" if cls == "d-ring" else "d-om"})"/>'
        return g + "</g>"
    out.append(ring("d-ring", 0)); out.append(ring("d-ring-tilt", tilt))
    out.append(f'<text x="{cx - 8}" y="{cy - r - 24}" text-anchor="end" class="d-note">z</text>')
    tx = cx + (r + 30) * math.sin(math.radians(tilt)); ty = cy - (r + 30) * math.cos(math.radians(tilt))
    out.append(f'<text x="{tx + 8:.1f}" y="{ty + 8:.1f}" class="d-om-text">learned axis, {tilt:.0f}° off</text>')
    lx = 420
    out.append(f'<line x1="{lx}" y1="96" x2="{lx + 34}" y2="96" class="d-ring"/><text x="{lx + 44}" y="100" class="d-note">conventional torus: L circles the z axis,</text>')
    out.append(f'<text x="{lx + 44}" y="116" class="d-note">L<tspan dy="3" font-size="9">z</tspan><tspan dy="-3"> is constant</tspan></text>')
    out.append(f'<line x1="{lx}" y1="150" x2="{lx + 34}" y2="150" class="d-ring-tilt"/><text x="{lx + 44}" y="154" class="d-note">learned torus: L circles a tilted axis,</text>')
    out.append(f'<text x="{lx + 44}" y="170" class="d-note">so L<tspan dy="3" font-size="9">z</tspan><tspan dy="-3"> varies; the component along</tspan></text>')
    out.append(f'<text x="{lx + 44}" y="186" class="d-note">the tilted axis is constant</text>')
    out.append(f'<text x="{lx}" y="232" class="d-note">Each point of a ring is one orbit: its plane is fixed,</text>')
    out.append(f'<text x="{lx}" y="248" class="d-note">and its L is conserved in both cases.</text>')
    out.append("</svg>")
    return "".join(out)


data = json.load(open(os.path.join(HERE, "note_data.json")))
page = open(os.path.join(HERE, "note.html")).read()
best = max((r for r in data["tilt"] if r["width"] == 128 and r["grown"]), key=lambda r: r["tilt_deg"])
blob = json.dumps(data, separators=(",", ":"))
out = (page.replace("__DATA__", blob).replace("__FIG_PHASE__", fig_phase()).replace("__FIG_ACTION__", fig_action())
       .replace("__FIG_SPHERE__", fig_sphere(best["tilt_deg"])))
open(os.path.join(HERE, "why_energy_suffices.html"), "w").write(out)
print(f"why_energy_suffices.html: {len(out) / 1e3:.0f} kB")
