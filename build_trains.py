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
HX, HY = 3.8, 8.5            # body half width / half length
ZB, ZT = 1.0, 9.0            # body bottom / top
BEV = 1.25                   # body corner rounding
Z_DARK, Z_STRIPE = 2.65, 3.4  # dark skirt below, colour stripe between

# part ids
BODY, UNDER, WHEEL, RIM, LENS, ROOFLAMP, BUMPER, GANGWAY, AC = range(9)

VARIANTS = [
    dict(name="TrainCab_Yellow", body=(0.99, 0.76, 0.25), stripe=(0.93, 0.40, 0.11), skirt=(0.27, 0.27, 0.29),
         lens=(1.0, 0.58, 0.14), front="cab", back="gangway", rake=0.0, roof_lamps=2, ac=False,
         side=[("win", -6.95, -5.75, 4.3, 7.3, 0), ("door", -4.9, -1.9), ("win", -0.9, 3.7, 4.4, 7.4, 1),
               ("win", 4.5, 7.3, 4.4, 7.4, 0)]),
    dict(name="TrainCar_Yellow", body=(0.99, 0.76, 0.25), stripe=(0.93, 0.40, 0.11), skirt=(0.27, 0.27, 0.29),
         lens=(1.0, 0.58, 0.14), front="gangway", back="gangway", rake=0.0, roof_lamps=0, ac=False,
         side=[("win", -7.4, -5.3, 4.4, 7.4, 0), ("door", -4.6, -1.8), ("win", -1.2, 1.2, 4.4, 7.4, 0),
               ("door", 1.8, 4.6), ("win", 5.3, 7.4, 4.4, 7.4, 0)]),
    dict(name="MetroCab_Blue", body=(0.25, 0.53, 0.88), stripe=(0.96, 0.96, 0.96), skirt=(0.33, 0.36, 0.42),
         lens=(1.0, 0.96, 0.80), front="cab", back="gangway", rake=0.0, roof_lamps=2, ac=True,
         door_col=(0.78, 0.80, 0.84),
         side=[("win", -6.95, -5.75, 4.3, 7.3, 0), ("door", -4.9, -2.1), ("win", -1.5, 1.5, 4.4, 7.4, 1),
               ("door", 2.1, 4.9), ("win", 5.5, 7.4, 4.4, 7.4, 0)]),
    dict(name="ExpressCab_Red", body=(0.86, 0.15, 0.13), stripe=(1.0, 0.80, 0.22), skirt=(0.22, 0.22, 0.25),
         lens=(1.0, 0.92, 0.70), front="cab", back="gangway", rake=2.0, roof_lamps=0, ac=False,
         door_col=(0.62, 0.62, 0.64),
         side=[("door", -5.4, -3.0), ("win", -2.4, 7.5, 4.7, 7.3, 4)]),
]

# ======================================================================== geometry helpers
def new_obj(name, bm, part, smooth=True):
    me = bpy.data.meshes.new(name); bm.to_mesh(me); bm.free()
    a = me.attributes.new("part", 'INT', 'FACE')
    a.data.foreach_set("value", [part] * len(me.polygons))
    for p in me.polygons: p.use_smooth = smooth
    o = bpy.data.objects.new(name, me); bpy.context.scene.collection.objects.link(o)
    return o

def rounded_box(name, x0, x1, y0, y1, z0, z1, off, segs, part, deform=None):
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    for v in bm.verts:
        v.co = Vector(((x0 + x1) / 2 + v.co.x * (x1 - x0), (y0 + y1) / 2 + v.co.y * (y1 - y0),
                       (z0 + z1) / 2 + v.co.z * (z1 - z0)))
    bmesh.ops.bevel(bm, geom=bm.verts[:] + bm.edges[:], offset=off, offset_type='OFFSET',
                    segments=segs, profile=0.5, affect='EDGES', clamp_overlap=True)
    if deform:
        for v in bm.verts: deform(v)
    return new_obj(name, bm, part)

def cylinder(name, center, axis, r1, r2, depth, segs, part, cap=True):
    bm = bmesh.new()
    rot = Vector((0, 0, 1)).rotation_difference(Vector(axis).normalized()).to_matrix().to_4x4()
    bmesh.ops.create_cone(bm, cap_ends=cap, cap_tris=False, segments=segs, radius1=r1, radius2=r2,
                          depth=depth, matrix=Matrix.Translation(center) @ rot)
    return new_obj(name, bm, part)

def build(v):
    objs = []
    rake = v["rake"]
    def rake_fn(vert):
        co = vert.co
        if rake and co.y < -4.5:
            t = max(0.0, min(1.0, (co.z - 3.9) / (ZT - 3.9)))
            w = max(0.0, min(1.0, (-co.y - 4.5) / 4.0))
            co.y += rake * t * w
    objs.append(rounded_box("body", -HX, HX, -HY, HY, ZB, ZT, BEV, 3, BODY, rake_fn))
    objs.append(rounded_box("under", -2.9, 2.9, -7.6, 7.6, 0.7, 1.6, 0.2, 1, UNDER))
    for y in (-6.5, -4.6, 4.6, 6.5):                                       # axles: wheel on each side
        objs.append(cylinder("axle", (0, y, 0.75), (1, 0, 0), 0.75, 0.75, 6.3, 10, WHEEL))
    for sgn in (-1, 1):                                                    # buffers / couplers
        objs.append(rounded_box("bumper", -1.3, 1.3, *sorted((sgn * (HY + 0.4), sgn * (HY - 0.8))), 0.85, 1.7, 0.12, 1, BUMPER))
    ends = [(-1, v["front"]), (1, v["back"])]
    for sgn, kind in ends:
        if kind == "gangway":                                              # rubber connector
            y0, y1 = (-HY - 0.22, -HY + 0.2) if sgn < 0 else (HY - 0.2, HY + 0.22)
            objs.append(rounded_box("gangway", -1.35, 1.35, y0, y1, 2.0, 7.7, 0.15, 1, GANGWAY))
    if v["front"] == "cab":
        for x in (-2.3, 2.3):                                            # headlights: rim + lens
            objs.append(cylinder("rim", (x, -HY - 0.08, 4.15), (0, -1, 0), 0.72, 0.72, 0.35, 12, RIM))
            objs.append(cylinder("lens", (x, -HY - 0.2, 4.15), (0, -1, 0), 0.54, 0.46, 0.32, 12, LENS))
        if v["roof_lamps"]:
            for x in (-0.85, 0.85):
                objs.append(cylinder("lamp", (x, -HY + 0.05, 8.1), (0, -1, 0), 0.44, 0.34, 0.5, 10, ROOFLAMP))
    if v["ac"]:
        objs.append(rounded_box("ac", -2.2, 2.2, -2.8, 2.8, ZT - 0.2, ZT + 0.75, 0.3, 2, AC))
    # join
    bpy.ops.object.select_all(action='DESELECT')
    for o in objs: o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    bpy.ops.object.join()
    obj = bpy.context.view_layer.objects.active
    obj.name = obj.data.name = v["name"]
    me = obj.data
    me.set_sharp_from_angle(angle=math.radians(35))
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
    dark_frame = np.array((0.24, 0.26, 0.28)); glass = np.array((0.20, 0.42, 0.47))
    glass_hi = np.array((0.36, 0.62, 0.66))
    door_col = np.array(v.get("door_col", (0.50, 0.49, 0.48)))

    body = ID == BODY
    put(body, v["body"])
    put(body & (z < Z_STRIPE), v["stripe"])
    put(body & (z < Z_DARK), v["skirt"])
    side = body & (np.abs(nx) > 0.6)
    # side features (same layout on both sides)
    for f in v["side"]:
        if f[0] == "door":
            y0, y1 = f[1], f[2]; cy, hw = (y0 + y1) / 2, (y1 - y0) / 2
            d = rrect(y, z, cy, 5.0, hw, 2.55, 0.25)
            put(side & (d < 0.12), dark_frame * 1.3)
            put(side & (d < 0.0), door_col)
            put(side & (d < 0.0) & (np.abs(y - cy) < 0.05), door_col * 0.6)          # centre split
            for wy in (cy - hw / 2, cy + hw / 2):                                   # door windows
                dw = rrect(y, z, wy, 5.9, hw / 2 - 0.45, 1.15, 0.25)
                put(side & (dw < 0.0), glass * 0.8)
        else:
            _, y0, y1, z0, z1, mull = f
            cy, cz, hw, hh = (y0 + y1) / 2, (z0 + z1) / 2, (y1 - y0) / 2, (z1 - z0) / 2
            d = rrect(y, z, cy, cz, hw, hh, 0.45)
            put(side & (d < 0.28), dark_frame)
            put(side & (d < 0.0), glass)
            hl = side & (d < 0.0) & (np.mod(y * 0.7 + z * 1.0, 3.0) < 0.35)            # glass glint
            put(hl, glass_hi)
            for k in range(1, mull + 1):
                my = y0 + (y1 - y0) * k / (mull + 1)
                put(side & (d < 0.0) & (np.abs(y - my) < 0.12), dark_frame)
    # ends
    for sgn, kind in ((-1, v["front"]), (1, v["back"])):
        end = body & (ny * sgn > 0.45)
        if kind == "cab":
            d = rrect(x, z, 0.0, 6.15, 2.75, 1.3, 0.7)
            put(end & (d < 0.38), dark_frame)
            put(end & (d < 0.0), glass * 1.1)
            put(end & (d < 0.0) & (np.abs(x * 0.55 - (z - 6.15) + 0.6) < 0.22), glass_hi * 1.15)
        else:
            d = rrect(x, z, 0.0, 5.2, 1.15, 2.7, 0.3)
            put(end & (d < 0.12), dark_frame)
            put(end & (d < 0.0), door_col)
            put(end & (rrect(x, z, 0.0, 6.5, 0.7, 0.85, 0.2) < 0.0), glass * 0.8)
    # other parts
    put(ID == UNDER, (0.19, 0.19, 0.21))
    wheel = ID == WHEEL
    put(wheel, (0.20, 0.20, 0.22))
    put(wheel & (np.abs(nx) > 0.7), (0.29, 0.29, 0.31))
    put(wheel & (np.abs(nx) > 0.7) & (np.hypot(np.minimum(np.abs(y - 6.5), np.minimum(np.abs(y - 4.6),
        np.minimum(np.abs(y + 4.6), np.abs(y + 6.5)))), z - 0.75) < 0.3), (0.48, 0.48, 0.50))
    put(ID == RIM, (0.24, 0.25, 0.27))
    put(ID == LENS, v["lens"])
    put(ID == ROOFLAMP, v["lens"])
    put(ID == BUMPER, (0.28, 0.28, 0.30))
    put(ID == GANGWAY, (0.24, 0.24, 0.26))
    put(ID == AC, (0.70, 0.72, 0.76))
    # baked soft lighting
    L = np.array((-0.35, -0.55, 0.76)); L /= np.linalg.norm(L)
    ndl = np.clip(nx * L[0] + ny * L[1] + nz * L[2], 0, 1)
    shade = 0.70 + 0.36 * ndl + 0.06 * nz
    shade *= np.where(ID == BODY, 0.88 + 0.12 * np.clip((z - ZB) / 3.0, 0, 1), 1.0)   # contact darkening
    col *= shade[..., None]
    # specular dot on lenses
    put((ID == LENS) & (ny < -0.8) & (np.hypot(x - np.sign(x) * 2.3 + 0.17, z - 4.33) < 0.14), (1, 1, 0.95))
    col = np.clip(col, 0, 1)
    return col

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
        cam.data.type = 'ORTHO'; cam.data.ortho_scale = 22
        scene.collection.objects.link(cam); scene.camera = cam
        target = Vector((0, 0, 4.5)); d = Vector((24, -26, 18))
        cam.location = target + d
        cam.rotation_euler = (-d).to_track_quat('-Z', 'Y').to_euler()
        scene.render.resolution_x = scene.render.resolution_y = 700
        scene.cycles.samples = 32; scene.view_settings.view_transform = 'Standard'
        scene.render.filepath = os.path.join(out, "preview.png")
        bpy.ops.render.render(write_still=True)

for v in VARIANTS:
    make_variant(v)
