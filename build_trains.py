"""Low-poly stylised train cars for Roblox (4 variants) — built with Blender 5.1.2 (bpy).

Run:  python build_trains.py [--preview]     (needs the `bpy` 5.1.2 module + Pillow + numpy)

Each variant gets ./Trains/<Name>/<Name>.fbx (+ _albedo.png, .blend).
Method: model the rounded body and protruding parts, then bake per-texel position / normal / part-id
maps with Cycles and paint windows, doors, stripes and baked lighting analytically into the texture.
"""
import bpy, bmesh, math, os, sys
import numpy as np
from mathutils import Vector, Matrix
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "Trains")
RENDER_PREVIEW = "--preview" in sys.argv
TEX = 1024

# Dimensions in studs. Front of the car faces -Y. Pivot = bottom centre (wheel contact).
# Proportions follow the reference: short, tall, chunky car (length ~1.7x width).
HX, HY = 3.8, 6.4            # body half width / half length
ZB, ZT = 0.9, 9.0            # body bottom / top
BEV_V, BEV_T = 0.9, 1.6      # rounding of the vertical corners / of the roof edge
Z_DARK, Z_STRIPE = 2.2, 2.95  # dark skirt below, colour stripe between
WS_CZ, WS_HW, WS_HH = 6.0, 2.75, 1.38     # windshield centre height, half width, half height
HL_Z, HL_X = 3.72, 2.42                   # headlights

# part ids
BODY, UNDER, WHEEL, RIM, LENS, ROOFLAMP, BUMPER, GANGWAY, AC, WSHIELD = range(10)

YEL = dict(body=(0.99, 0.77, 0.27), stripe=(0.92, 0.40, 0.11), skirt=(0.27, 0.27, 0.29), lens=(1.0, 0.58, 0.14))
VARIANTS = [
    dict(name="TrainCab_Yellow", **YEL, front="cab", back="gangway", rake=0.0, roof_lamps=True, ac=False,
         side=[("win", -5.35, -4.45, 4.35, 7.05, 0), ("door", -3.85, -0.95), ("win", 0.0, 5.1, 4.6, 7.25, 1)]),
    dict(name="TrainCar_Yellow", **YEL, front="gangway", back="gangway", rake=0.0, roof_lamps=False, ac=False,
         side=[("win", -5.3, -2.5, 4.6, 7.25, 0), ("door", -1.45, 1.45), ("win", 2.5, 5.3, 4.6, 7.25, 0)]),
    dict(name="MetroCab_Blue", body=(0.25, 0.53, 0.88), stripe=(0.96, 0.96, 0.96), skirt=(0.32, 0.35, 0.41),
         lens=(1.0, 0.96, 0.80), door_col=(0.76, 0.78, 0.82),
         front="cab", back="gangway", rake=0.0, roof_lamps=True, ac=True,
         side=[("win", -5.35, -4.45, 4.35, 7.05, 0), ("door", -3.85, -0.95), ("win", 0.0, 5.1, 4.6, 7.25, 2)]),
    dict(name="ExpressCab_Red", body=(0.87, 0.16, 0.14), stripe=(1.0, 0.80, 0.22), skirt=(0.22, 0.22, 0.25),
         lens=(1.0, 0.93, 0.72), door_col=(0.62, 0.62, 0.64),
         front="cab", back="gangway", rake=1.3, roof_lamps=False, ac=False,
         side=[("door", -4.3, -1.9), ("win", -1.2, 5.3, 4.7, 7.2, 2)]),
]

# ======================================================================== geometry helpers
def new_obj(name, bm, part, smooth=True, deform=None):
    if deform:
        for vt in bm.verts: deform(vt)
    me = bpy.data.meshes.new(name); bm.to_mesh(me); bm.free()
    a = me.attributes.new("part", 'INT', 'FACE')
    a.data.foreach_set("value", [part] * len(me.polygons))
    for p in me.polygons: p.use_smooth = smooth
    o = bpy.data.objects.new(name, me); bpy.context.scene.collection.objects.link(o)
    return o

def box_bm(x0, x1, y0, y1, z0, z1):
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    for vt in bm.verts:
        vt.co = Vector(((x0 + x1) / 2 + vt.co.x * (x1 - x0), (y0 + y1) / 2 + vt.co.y * (y1 - y0),
                        (z0 + z1) / 2 + vt.co.z * (z1 - z0)))
    return bm

def rounded_box(name, x0, x1, y0, y1, z0, z1, off, segs, part, deform=None):
    bm = box_bm(x0, x1, y0, y1, z0, z1)
    bmesh.ops.bevel(bm, geom=bm.verts[:] + bm.edges[:], offset=off, offset_type='OFFSET',
                    segments=segs, profile=0.5, affect='EDGES', clamp_overlap=True)
    return new_obj(name, bm, part, deform=deform)

def body_shell(deform):
    """Loft of rounded-rectangle rings: vertical sides with corner radius BEV_V, a quarter-round roof edge
    of radius BEV_T and a small bottom chamfer. Built ring by ring so the corners never fold over."""
    bm = bmesh.new()
    def ring(inset, z, segs=3):
        hx, hy = HX - inset, HY - inset
        rc = min(BEV_V, hx - 0.05)
        pts = []
        for (sx, sy), a0 in (((1, 1), 0), ((-1, 1), 90), ((-1, -1), 180), ((1, -1), 270)):
            cx, cy = sx * (hx - rc), sy * (hy - rc)
            for k in range(segs + 1):
                a = math.radians(a0 + 90 * k / segs)
                pts.append(bm.verts.new((cx + rc * math.cos(a), cy + rc * math.sin(a), z)))
        return pts
    levels = [(0.3, ZB), (0.0, ZB + 0.3), (0.0, ZT - BEV_T)]
    for k in range(1, 5):                                   # quarter-round roof edge, 4 steps
        t = math.radians(90 * k / 4)
        levels.append((BEV_T * (1 - math.cos(t)), ZT - BEV_T + BEV_T * math.sin(t)))
    rings = [ring(i, z) for i, z in levels]
    n = len(rings[0])
    for r0, r1 in zip(rings, rings[1:]):
        for j in range(n):
            bm.faces.new((r0[j], r0[(j + 1) % n], r1[(j + 1) % n], r1[j]))
    caps = [bm.faces.new(rings[0][::-1]), bm.faces.new(rings[-1])]
    bmesh.ops.poke(bm, faces=caps)
    return new_obj("body", bm, BODY, deform=deform)

def rrect_slab(name, cx, cz, hw, hh, r, y_back, y_front, part, deform=None, segs=3):
    """Rounded-rectangle plate facing -Y (front cap + rim, back left open: it sits on the body)."""
    bm = bmesh.new()
    pts = []
    for (sx, sz), a0 in (((1, 1), 0), ((-1, 1), 90), ((-1, -1), 180), ((1, -1), 270)):
        ccx, ccz = cx + sx * (hw - r), cz + sz * (hh - r)
        for k in range(segs + 1):
            a = math.radians(a0 + 90 * k / segs)
            pts.append((ccx + r * math.cos(a), ccz + r * math.sin(a)))
    back = [bm.verts.new((x, y_back, z)) for x, z in pts]
    front = [bm.verts.new((x, y_front, z)) for x, z in pts]
    n = len(pts)
    for i in range(n):
        j = (i + 1) % n
        bm.faces.new((back[i], back[j], front[j], front[i]))
    bm.faces.new(front)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    return new_obj(name, bm, part, deform=deform)

def cylinder(name, center, axis, r1, r2, depth, segs, part, cap=True):
    bm = bmesh.new()
    rot = Vector((0, 0, 1)).rotation_difference(Vector(axis).normalized()).to_matrix().to_4x4()
    bmesh.ops.create_cone(bm, cap_ends=cap, cap_tris=False, segments=segs, radius1=r1, radius2=r2,
                          depth=depth, matrix=Matrix.Translation(center) @ rot)
    return new_obj(name, bm, part)

def make_rake(rake):
    """Lean the upper front back by `rake` studs (shear, so flat parts stay flat)."""
    def fn(vt):
        co = vt.co
        if rake and co.y < -HY + 3.0:
            t = max(0.0, min(1.0, (co.z - 4.2) / (ZT - 4.2)))
            w = max(0.0, min(1.0, (-co.y - (HY - 3.0)) / 2.0))
            co.y += rake * t * w
    return fn

def build(v):
    objs = []
    rk = make_rake(v["rake"])
    objs.append(body_shell(rk))
    objs.append(rounded_box("under", -3.1, 3.1, -5.6, 5.6, 0.45, 1.2, 0.15, 1, UNDER))
    for y in (-4.6, -3.0, 3.0, 4.6):                                       # axles: a wheel on each side
        objs.append(cylinder("axle", (0, y, 0.78), (1, 0, 0), 0.78, 0.78, 6.9, 10, WHEEL))
    for sgn in (-1, 1):                                                    # flat buffer plate
        objs.append(rounded_box("bumper", -1.9, 1.9, *sorted((sgn * (HY + 0.55), sgn * (HY - 0.6))),
                                0.55, 1.05, 0.12, 1, BUMPER))
    for sgn, kind in ((-1, v["front"]), (1, v["back"])):
        if kind == "gangway":                                              # rubber connector
            y0, y1 = (-HY - 0.22, -HY + 0.2) if sgn < 0 else (HY - 0.2, HY + 0.22)
            objs.append(rounded_box("gangway", -1.2, 1.2, y0, y1, 2.0, 7.3, 0.15, 1, GANGWAY))
    if v["front"] == "cab":
        # raised windshield frame (glass is painted on its face)
        objs.append(rrect_slab("windshield", 0.0, WS_CZ, WS_HW, WS_HH, 0.62, -HY + 0.1, -HY - 0.13,
                               WSHIELD, deform=rk))
        for x in (-HL_X, HL_X):                                            # headlights: rim + lens
            objs.append(cylinder("rim", (x, -HY - 0.1, HL_Z), (0, -1, 0), 0.74, 0.70, 0.4, 12, RIM))
            objs.append(cylinder("lens", (x, -HY - 0.3, HL_Z), (0, -1, 0), 0.56, 0.44, 0.22, 12, LENS))
        if v["roof_lamps"]:
            # sit on the curved roof edge, pointing along the surface normal
            zc = ZT - BEV_T; yc = -HY + BEV_T; a = math.radians(28)       # 28 deg up the curve
            p = Vector((0, yc - BEV_T * math.cos(a), zc + BEV_T * math.sin(a)))
            nrm = Vector((0, -math.cos(a), math.sin(a)))
            for x in (-0.85, 0.85):
                objs.append(cylinder("lamp", Vector((x, 0, 0)) + p + nrm * 0.12, nrm, 0.56, 0.30, 0.55, 10, ROOFLAMP))
    if v["ac"]:
        objs.append(rounded_box("ac", -2.0, 2.0, -2.4, 2.4, ZT - 0.2, ZT + 0.7, 0.3, 2, AC))
    bpy.ops.object.select_all(action='DESELECT')
    for o in objs: o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    bpy.ops.object.join()
    obj = bpy.context.view_layer.objects.active
    obj.name = obj.data.name = v["name"]
    me = obj.data
    me.set_sharp_from_angle(angle=math.radians(40))
    bm = bmesh.new(); bm.from_mesh(me)
    bmesh.ops.triangulate(bm, faces=bm.faces[:])
    bm.to_mesh(me); bm.free()
    return obj

# ======================================================================== texture painting
def rrect(a, b, cx, cz, hw, hh, r):
    """signed distance to a rounded rectangle in the (a, b) plane (negative inside)."""
    qx = np.abs(a - cx) - hw + r; qz = np.abs(b - cz) - hh + r
    return np.hypot(np.maximum(qx, 0), np.maximum(qz, 0)) + np.minimum(np.maximum(qx, qz), 0) - r

def paint(v, P, N, ID, cover):
    x, y, z = P[..., 0], P[..., 1], P[..., 2]
    nx, ny, nz = N[..., 0], N[..., 1], N[..., 2]
    H, W = ID.shape
    col = np.zeros((H, W, 3), np.float32)
    def put(mask, c): col[mask] = c
    frame = np.array((0.26, 0.28, 0.30)); glass = np.array((0.22, 0.50, 0.56))
    glass_dk = np.array((0.15, 0.33, 0.37)); glass_hi = np.array((0.36, 0.66, 0.70))
    door_col = np.array(v.get("door_col", (0.47, 0.46, 0.45)))
    # un-rake z for the front so painted shapes follow the leaned windshield
    body = ID == BODY
    put(body, v["body"])
    put(body & (z < Z_STRIPE), v["stripe"])
    put(body & (z < Z_DARK), v["skirt"])
    side = body & (np.abs(nx) > 0.6)
    for f in v["side"]:
        if f[0] == "door":                                  # recessed door: frame, two leaves, windows
            y0, y1 = f[1], f[2]; cy, hw = (y0 + y1) / 2, (y1 - y0) / 2
            top, bot = 7.35, Z_DARK
            d = rrect(y, z, cy, (top + bot) / 2, hw, (top - bot) / 2, 0.22)
            put(side & (d < 0.16), frame * 1.25)
            put(side & (d < 0.0), door_col)
            put(side & (d < 0.0) & (d > -0.12) & (z > (top + bot) / 2), door_col * 0.78)   # inner shadow
            put(side & (d < 0.0) & (np.abs(y - cy) < 0.06), door_col * 0.55)              # centre split
            for wy in (cy - hw / 2, cy + hw / 2):
                dw = rrect(y, z, wy, 5.75, hw / 2 - 0.42, 1.25, 0.32)
                put(side & (dw < 0.08), frame * 1.1)
                put(side & (dw < 0.0), glass_dk)
        else:
            _, y0, y1, z0, z1, mull = f
            cy, cz, hw, hh = (y0 + y1) / 2, (z0 + z1) / 2, (y1 - y0) / 2, (z1 - z0) / 2
            d = rrect(y, z, cy, cz, hw, hh, 0.42)
            put(side & (d < 0.3), frame)
            put(side & (d < 0.0), glass_dk)
            put(side & (d < 0.0) & (z > cz + 0.2 * hh), (glass_dk + glass) / 2)
            for k in range(1, mull + 1):
                my = y0 + (y1 - y0) * k / (mull + 1)
                put(side & (d < 0.0) & (np.abs(y - my) < 0.13), frame)
    for sgn, kind in ((-1, v["front"]), (1, v["back"])):  # gangway door on connector ends
        end = body & (ny * sgn > 0.45)
        if kind == "gangway":
            d = rrect(x, z, 0.0, 4.9, 1.1, 2.6, 0.3)
            put(end & (d < 0.12), frame)
            put(end & (d < 0.0), door_col)
            put(end & (rrect(x, z, 0.0, 6.1, 0.65, 0.85, 0.22) < 0.0), glass_dk)
    # windshield plate: dark frame rim, teal glass with a diagonal reflection
    ws = ID == WSHIELD
    zl = z                                  # the rake shear moves y only, so z still maps the glass
    put(ws, frame)
    g = ws & (ny < -0.6) & (rrect(x, zl, 0.0, WS_CZ, WS_HW - 0.34, WS_HH - 0.34, 0.42) < 0.0)
    put(g, glass)
    put(g & (zl > WS_CZ + 0.15 * (x + 1.0)), glass * 1.08)
    put(g & (np.abs((zl - WS_CZ) - 0.62 * (x + 0.9)) < 0.28), glass_hi)
    put(g & (np.abs((zl - WS_CZ) - 0.62 * (x + 0.9) - 0.6) < 0.08), glass_hi)
    # other parts
    put(ID == UNDER, (0.18, 0.18, 0.20))
    put(ID == WHEEL, (0.19, 0.19, 0.21))
    hub = np.full(x.shape, 9.0)
    for yy in (-4.6, -3.0, 3.0, 4.6):
        hub = np.minimum(hub, np.hypot(y - yy, z - 0.78))
    put((ID == WHEEL) & (np.abs(nx) > 0.7), (0.27, 0.27, 0.29))
    put((ID == WHEEL) & (np.abs(nx) > 0.7) & (hub < 0.32), (0.42, 0.42, 0.44))
    put(ID == RIM, (0.25, 0.26, 0.28))
    put(ID == LENS, v["lens"])
    put(ID == ROOFLAMP, v["lens"])
    put(ID == BUMPER, (0.27, 0.27, 0.29))
    put(ID == GANGWAY, (0.33, 0.33, 0.35))
    put(ID == AC, (0.70, 0.72, 0.76))
    # baked soft studio lighting (key from upper front-left + sky fill)
    L = np.array((-0.35, -0.6, 0.72)); L /= np.linalg.norm(L)
    ndl = np.clip(nx * L[0] + ny * L[1] + nz * L[2], 0, 1)
    shade = 0.74 + 0.30 * ndl + 0.07 * nz
    shade *= np.where(ID == BODY, 0.9 + 0.1 * np.clip((z - ZB) / 2.5, 0, 1), 1.0)
    col *= shade[..., None]
    # glossy highlight dots on the lenses / roof lamps
    lens_hi = (ID == LENS) & (ny < -0.6) & (np.hypot(np.abs(x) - HL_X + 0.16, z - HL_Z - 0.18) < 0.14)
    put(lens_hi, (1.0, 0.97, 0.88))
    put((ID == ROOFLAMP) & (nz > 0.75), np.minimum(np.array(v["lens"]) * 1.25, 1.0))
    return np.clip(col, 0, 1)

# ======================================================================== bake helpers
def bake_pass(obj, mat, img, build_input):
    nt = mat.node_tree
    emit = nt.nodes["Emission"]
    for l in list(emit.inputs["Color"].links): nt.links.remove(l)
    build_input(nt, emit.inputs["Color"])
    nt.nodes["Bake"].image = img
    nt.nodes.active = nt.nodes["Bake"]
    bpy.ops.object.bake(type='EMIT', margin=4, use_clear=True)
    a = np.empty(TEX * TEX * 4, np.float32); img.pixels.foreach_get(a)
    return a.reshape(TEX, TEX, 4)[::-1]          # flip so row 0 = top (matches PNG)

def make_variant(v):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    out = os.path.join(ROOT, v["name"]); os.makedirs(out, exist_ok=True)
    obj = build(v)
    me = obj.data
    bpy.context.view_layer.objects.active = obj; obj.select_set(True)
    me.uv_layers.new(name="UVMap")
    bpy.ops.object.mode_set(mode='EDIT'); bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.smart_project(angle_limit=math.radians(50), island_margin=0.008, area_weight=0.0,
                             correct_aspect=True, scale_to_bounds=True)
    bpy.ops.object.mode_set(mode='OBJECT')

    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'; scene.cycles.device = 'CPU'; scene.cycles.samples = 1
    mat = bpy.data.materials.new(f"{v['name']}_Mat"); nt = mat.node_tree; nt.nodes.clear()
    o = nt.nodes.new("ShaderNodeOutputMaterial")
    e = nt.nodes.new("ShaderNodeEmission"); e.name = "Emission"
    nt.links.new(e.outputs[0], o.inputs["Surface"])
    t = nt.nodes.new("ShaderNodeTexImage"); t.name = "Bake"
    me.materials.append(mat)

    def fimg(n):
        im = bpy.data.images.new(n, TEX, TEX, alpha=True, float_buffer=True)
        im.colorspace_settings.name = 'Non-Color'; return im
    lo, hi = Vector((-6, -12, -1)), Vector((6, 12, 12))
    def pos_in(nt, sock):
        tc = nt.nodes.new("ShaderNodeTexCoord")
        vm = nt.nodes.new("ShaderNodeVectorMath"); vm.operation = 'MULTIPLY_ADD'   # (p - lo) / (hi - lo)
        span = hi - lo
        vm.inputs[1].default_value = (1 / span.x, 1 / span.y, 1 / span.z)
        vm.inputs[2].default_value = (-lo.x / span.x, -lo.y / span.y, -lo.z / span.z)
        nt.links.new(tc.outputs["Object"], vm.inputs[0]); nt.links.new(vm.outputs[0], sock)
    def nrm_in(nt, sock):
        g = nt.nodes.new("ShaderNodeNewGeometry")
        vm = nt.nodes.new("ShaderNodeVectorMath"); vm.operation = 'MULTIPLY_ADD'
        vm.inputs[1].default_value = (0.5, 0.5, 0.5); vm.inputs[2].default_value = (0.5, 0.5, 0.5)
        nt.links.new(g.outputs["Normal"], vm.inputs[0]); nt.links.new(vm.outputs[0], sock)
    def id_in(nt, sock):
        at = nt.nodes.new("ShaderNodeAttribute"); at.attribute_name = "part"; at.attribute_type = 'GEOMETRY'
        m = nt.nodes.new("ShaderNodeMath"); m.operation = 'DIVIDE'; m.inputs[1].default_value = 16.0
        nt.links.new(at.outputs["Fac"], m.inputs[0]); nt.links.new(m.outputs[0], sock)

    Pm = bake_pass(obj, mat, fimg("pos"), pos_in)
    Nm = bake_pass(obj, mat, fimg("nrm"), nrm_in)
    Im = bake_pass(obj, mat, fimg("id"), id_in)
    cover = Pm[..., 3] > 0.5
    P = np.array(lo) + Pm[..., :3] * (np.array(hi) - np.array(lo))
    Nn = Nm[..., :3] * 2 - 1
    Nn /= np.maximum(np.linalg.norm(Nn, axis=-1, keepdims=True), 1e-6)
    ID = np.rint(Im[..., 0] * 16).astype(int)
    rgb = paint(v, P, Nn, ID, cover)
    # pad every island outward with its own colours so filtering never pulls in empty texels
    known = cover.copy()
    for _ in range(24):
        acc = np.zeros_like(rgb); cnt = np.zeros(known.shape, np.float32)
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            k = np.roll(known, (dy, dx), (0, 1)); acc += np.roll(rgb, (dy, dx), (0, 1)) * k[..., None]; cnt += k
        grow = (~known) & (cnt > 0)
        rgb[grow] = acc[grow] / cnt[grow][:, None]; known |= grow
    png = os.path.join(out, f"{v['name']}_albedo.png")
    Image.fromarray((rgb * 255 + 0.5).astype(np.uint8)).save(png)

    # final material
    nt.nodes.clear()
    o = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled"); bsdf.inputs["Roughness"].default_value = 0.6
    tex = nt.nodes.new("ShaderNodeTexImage"); tex.image = bpy.data.images.load(png); tex.image.pack()
    nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(bsdf.outputs["BSDF"], o.inputs["Surface"])
    for n in ("pos", "nrm", "id"): bpy.data.images.remove(bpy.data.images[n])
    me.attributes.remove(me.attributes["part"])

    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(out, f"{v['name']}.blend"))
    bpy.ops.export_scene.fbx(
        filepath=os.path.join(out, f"{v['name']}.fbx"), use_selection=True, object_types={'MESH'},
        apply_scale_options='FBX_SCALE_ALL', axis_forward='-Z', axis_up='Y',
        mesh_smooth_type='OFF', use_tspace=False, path_mode='COPY', embed_textures=True, bake_anim=False)
    print(f"{v['name']}: TRIANGLES={len(me.polygons)} dims={tuple(round(d, 2) for d in obj.dimensions)}")

    if RENDER_PREVIEW:
        from mathutils import Euler
        world = bpy.data.worlds.new("W"); scene.world = world
        bg = world.node_tree.nodes["Background"]
        bg.inputs[0].default_value = (1, 1, 1, 1); bg.inputs[1].default_value = 0.9
        sun = bpy.data.objects.new("Sun", bpy.data.lights.new("Sun", 'SUN'))
        sun.data.energy = 2.0; sun.rotation_euler = Euler((math.radians(40), 0, math.radians(-30)))
        scene.collection.objects.link(sun)
        cam = bpy.data.objects.new("Cam", bpy.data.cameras.new("Cam"))
        cam.data.type = 'ORTHO'; cam.data.ortho_scale = 18
        scene.collection.objects.link(cam); scene.camera = cam
        target = Vector((0, 0, 4.4)); d = Vector((20, -27, 17))
        cam.location = target + d
        cam.rotation_euler = (-d).to_track_quat('-Z', 'Y').to_euler()
        scene.render.resolution_x = scene.render.resolution_y = 700
        scene.cycles.samples = 32; scene.view_settings.view_transform = 'Standard'
        scene.render.filepath = os.path.join(out, "preview.png")
        bpy.ops.render.render(write_still=True)

for v in VARIANTS:
    make_variant(v)
