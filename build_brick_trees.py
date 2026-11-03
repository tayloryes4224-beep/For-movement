"""Classic Roblox 'brick' trees with baked studs — built with Blender 5.1.2 (bpy).

Run:  python build_brick_trees.py [--preview]     (needs the `bpy` 5.1.2 module + Pillow + numpy)

Each tree is made only of axis-aligned blocks snapped to the 1-stud grid. A Cycles bake writes
per-texel world position / normal / block-id maps, and numpy paints the classic stud pattern
(1 stud per grid cell, on every face), block outlines and soft lighting into one texture.
Outputs ./BrickTrees/<Name>/<Name>.fbx (+ _albedo.png, .blend, preview.png).
"""
import bpy, bmesh, math, os, sys, random
import numpy as np
from mathutils import Vector
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "BrickTrees")
RENDER_PREVIEW = "--preview" in sys.argv
TEX = 1024

WOOD, LEAF = 0, 1
WOOD_COL = (0.50, 0.31, 0.19)
LEAF_COL = (0.46, 0.82, 0.30)

def W(x0, x1, y0, y1, z0, z1): return (WOOD, x0, x1, y0, y1, z0, z1)
def G(x0, x1, y0, y1, z0, z1): return (LEAF, x0, x1, y0, y1, z0, z1)

TREES = {
    # big tree from the reference: tall square trunk, two tiers of branch slabs carrying leaf blocks, stepped crown
    "BrickOak": [
        W(-3, 3, -3, 3, 0, 26), W(-4, 4, -4, 4, 0, 2),
        # tier 1: four arms + leaf pads
        W(3, 12, -2, 2, 12, 14), W(-12, -3, -1, 3, 12, 14), W(-2, 2, 3, 11, 12, 14), W(-3, 1, -12, -3, 12, 14),
        G(7, 16, -5, 4, 14, 17), G(-16, -8, -3, 6, 14, 17), G(-4, 5, 8, 15, 14, 17), G(-6, 3, -16, -8, 14, 17),
        G(9, 14, -3, 2, 17, 18), G(-14, -10, -1, 4, 17, 18),
        # tier 2: rotated arms + quadrant leaf pads
        W(3, 10, 0, 3, 19, 21), W(-10, -3, -3, 0, 19, 21), W(-3, 0, 3, 10, 19, 21), W(0, 3, -10, -3, 19, 21),
        G(4, 13, 1, 10, 21, 24), G(-13, -4, -10, -1, 21, 24), G(-10, -1, 4, 13, 21, 24), G(1, 10, -13, -4, 21, 24),
        # crown
        W(-8, 8, -2, 2, 25, 27), W(-2, 2, -8, 8, 25, 27),
        G(-11, 2, -9, 1, 27, 30), G(-2, 11, -1, 10, 27, 30), G(1, 10, -10, 0, 28, 31), G(-9, 0, 0, 10, 28, 31),
        G(-6, 6, -6, 6, 31, 34), G(-4, 2, -3, 4, 34, 37), G(0, 5, -5, 0, 34, 36),
    ],
    # the small stacked tree on the left of the reference
    "BrickPine": [
        W(-2, 2, -2, 2, 0, 7), W(-4, 4, -3, 3, 4, 7),
        G(-7, 7, -6, 6, 7, 10), G(5, 9, -2, 3, 8, 11), G(-9, -5, -3, 2, 8, 11),
        G(-5, 6, -5, 5, 10, 13), G(-6, -3, 1, 6, 11, 14),
        G(-4, 4, -4, 4, 13, 16), G(-3, 3, -2, 3, 16, 19), G(-2, 2, -2, 2, 19, 21),
    ],
    # a smaller round-ish oak
    "BrickOakSmall": [
        W(-2, 2, -2, 2, 0, 11), W(-3, 3, -3, 3, 0, 1),
        W(-7, 7, -1, 1, 8, 10), W(-1, 1, -7, 7, 8, 10),
        G(-9, 9, -8, 8, 10, 13), G(6, 11, -3, 4, 9, 12), G(-11, -6, -4, 3, 9, 12), G(-3, 4, -11, -6, 9, 12),
        G(-6, 6, -6, 7, 13, 16), G(-7, -3, -2, 4, 14, 16),
        G(-3, 4, -4, 3, 16, 18), G(-1, 2, -1, 2, 18, 19),
    ],
}

# ======================================================================== geometry
def inside_any(p, boxes, skip):
    for i, (_, x0, x1, y0, y1, z0, z1) in enumerate(boxes):
        if i != skip and x0 < p.x < x1 and y0 < p.y < y1 and z0 < p.z < z1:
            return True
    return False

def build(name, boxes):
    bm = bmesh.new()
    bid = bm.faces.layers.int.new("block")
    for i, (kind, x0, x1, y0, y1, z0, z1) in enumerate(boxes):
        V = {}
        for xi, x in enumerate((x0, x1)):
            for yi, y in enumerate((y0, y1)):
                for zi, z in enumerate((z0, z1)):
                    V[xi, yi, zi] = bm.verts.new((x, y, z))
        quads = [((0, 0, 0), (0, 1, 0), (0, 1, 1), (0, 0, 1)), ((1, 0, 0), (1, 0, 1), (1, 1, 1), (1, 1, 0)),
                 ((0, 0, 0), (0, 0, 1), (1, 0, 1), (1, 0, 0)), ((0, 1, 0), (1, 1, 0), (1, 1, 1), (0, 1, 1)),
                 ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)), ((0, 0, 1), (0, 1, 1), (1, 1, 1), (1, 0, 1))]
        for q in quads:
            vs = [V[k] for k in q]
            c = sum((v.co for v in vs), Vector()) / 4
            n = (vs[1].co - vs[0].co).cross(vs[2].co - vs[0].co).normalized()
            probes = [c] + [v.co + (c - v.co) * 0.02 for v in vs] + [(vs[k].co + vs[(k + 1) % 4].co) / 2 * 0.98 + c * 0.02 for k in range(4)]
            if all(inside_any(q + n * 0.01, boxes, i) for q in probes):   # fully buried face: drop it
                continue
            f = bm.faces.new(vs); f[bid] = i
    bm.normal_update()
    bmesh.ops.remove_doubles(bm, verts=bm.verts[:], dist=1e-5)
    bmesh.ops.triangulate(bm, faces=bm.faces[:])
    me = bpy.data.meshes.new(name); bm.to_mesh(me); bm.free()
    for p in me.polygons: p.use_smooth = False
    obj = bpy.data.objects.new(name, me); bpy.context.scene.collection.objects.link(obj)
    return obj

# ======================================================================== stud painting
def paint(boxes, P, N, ID, seed):
    rng = random.Random(seed)
    tone = np.array([rng.uniform(0.93, 1.07) for _ in boxes] + [1.0], np.float32)
    kind = np.array([b[0] for b in boxes] + [0])
    B = np.array([b[1:] for b in boxes] + [(0,) * 6], np.float32)
    ID = np.clip(ID, 0, len(boxes))
    x, y, z = P[..., 0], P[..., 1], P[..., 2]
    nx, ny, nz = N[..., 0], N[..., 1], N[..., 2]
    ax = np.argmax(np.abs(N), axis=-1)                  # dominant axis of each texel's face
    u = np.where(ax == 0, y, x)
    v = np.where(ax == 2, y, z)
    # base colour per block
    base = np.where((kind[ID] == LEAF)[..., None], np.array(LEAF_COL), np.array(WOOD_COL)) * tone[ID][..., None]
    # lighting: bright tops, mid sides, dark bottoms
    L = np.array((-0.45, -0.55, 0.70)); L /= np.linalg.norm(L)
    ndl = np.clip(nx * L[0] + ny * L[1] + nz * L[2], 0, 1)
    shade = 0.70 + 0.40 * ndl + 0.10 * nz
    # studs: one per 1x1 cell, rim lit from the upper-left of the face
    fu = u - np.floor(u) - 0.5; fv = v - np.floor(v) - 0.5
    r = np.hypot(fu, fv) + 1e-6
    lu, lv = -0.7071, 0.7071
    side = (fu * lu + fv * lv) / r                      # +1 on the lit side of the stud, -1 opposite
    rim = (r > 0.23) & (r < 0.34)
    stud = shade * np.where(rim, 1.0 + 0.38 * side, np.where(r <= 0.23, 1.07, 1.0))
    stud *= np.where((r >= 0.34) & (r < 0.40), 1.0 - 0.22 * np.clip(-side, 0, 1), 1.0)   # tiny drop shadow
    # dark outline along every block edge
    x0, x1, y0, y1, z0, z1 = (B[ID][..., k] for k in range(6))
    du = np.where(ax == 0, np.minimum(y - y0, y1 - y), np.minimum(x - x0, x1 - x))
    dv = np.where(ax == 2, np.minimum(y - y0, y1 - y), np.minimum(z - z0, z1 - z))
    edge = np.minimum(du, dv)
    stud *= np.where(edge < 0.07, 0.72, np.where(edge < 0.14, 0.9, 1.0))
    return np.clip(base * stud[..., None], 0, 1)

# ======================================================================== bake + export
def bake_pass(mat, img, build_input):
    nt = mat.node_tree; emit = nt.nodes["Emission"]
    for l in list(emit.inputs["Color"].links): nt.links.remove(l)
    build_input(nt, emit.inputs["Color"])
    nt.nodes["Bake"].image = img; nt.nodes.active = nt.nodes["Bake"]
    bpy.ops.object.bake(type='EMIT', margin=2, use_clear=True)
    a = np.empty(TEX * TEX * 4, np.float32); img.pixels.foreach_get(a)
    return a.reshape(TEX, TEX, 4)[::-1]

def make_tree(name, boxes):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    out = os.path.join(ROOT, name); os.makedirs(out, exist_ok=True)
    obj = build(name, boxes); me = obj.data
    bpy.context.view_layer.objects.active = obj; obj.select_set(True)
    me.uv_layers.new(name="UVMap")
    bpy.ops.object.mode_set(mode='EDIT'); bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.smart_project(angle_limit=math.radians(30), island_margin=0.004, area_weight=0.0,
                             correct_aspect=True, scale_to_bounds=True)
    bpy.ops.object.mode_set(mode='OBJECT')

    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'; scene.cycles.device = 'CPU'; scene.cycles.samples = 1
    mat = bpy.data.materials.new(f"{name}_Mat"); nt = mat.node_tree; nt.nodes.clear()
    o = nt.nodes.new("ShaderNodeOutputMaterial")
    e = nt.nodes.new("ShaderNodeEmission"); e.name = "Emission"
    nt.links.new(e.outputs[0], o.inputs["Surface"])
    t = nt.nodes.new("ShaderNodeTexImage"); t.name = "Bake"
    me.materials.append(mat)
    def fimg(n):
        im = bpy.data.images.new(n, TEX, TEX, alpha=True, float_buffer=True)
        im.colorspace_settings.name = 'Non-Color'; return im
    lo, hi = Vector((-40, -40, -2)), Vector((40, 40, 60))
    def pos_in(nt, sock):
        tc = nt.nodes.new("ShaderNodeTexCoord")
        vm = nt.nodes.new("ShaderNodeVectorMath"); vm.operation = 'MULTIPLY_ADD'
        sp = hi - lo
        vm.inputs[1].default_value = (1 / sp.x, 1 / sp.y, 1 / sp.z)
        vm.inputs[2].default_value = (-lo.x / sp.x, -lo.y / sp.y, -lo.z / sp.z)
        nt.links.new(tc.outputs["Object"], vm.inputs[0]); nt.links.new(vm.outputs[0], sock)
    def nrm_in(nt, sock):
        g = nt.nodes.new("ShaderNodeNewGeometry")
        vm = nt.nodes.new("ShaderNodeVectorMath"); vm.operation = 'MULTIPLY_ADD'
        vm.inputs[1].default_value = (0.5, 0.5, 0.5); vm.inputs[2].default_value = (0.5, 0.5, 0.5)
        nt.links.new(g.outputs["True Normal"], vm.inputs[0]); nt.links.new(vm.outputs[0], sock)
    def id_in(nt, sock):
        at = nt.nodes.new("ShaderNodeAttribute"); at.attribute_name = "block"; at.attribute_type = 'GEOMETRY'
        m = nt.nodes.new("ShaderNodeMath"); m.operation = 'DIVIDE'; m.inputs[1].default_value = 256.0
        nt.links.new(at.outputs["Fac"], m.inputs[0]); nt.links.new(m.outputs[0], sock)
    Pm = bake_pass(mat, fimg("pos"), pos_in)
    Nm = bake_pass(mat, fimg("nrm"), nrm_in)
    Im = bake_pass(mat, fimg("id"), id_in)
    cover = Pm[..., 3] > 0.5
    P = np.array(lo) + Pm[..., :3] * (np.array(hi) - np.array(lo))
    Nn = Nm[..., :3] * 2 - 1
    ID = np.rint(Im[..., 0] * 256).astype(int)
    rgb = paint(boxes, P, Nn, ID, seed=len(name)).astype(np.float32)
    known = cover.copy()                                # pad islands with their own colours
    for _ in range(16):
        acc = np.zeros_like(rgb); cnt = np.zeros(known.shape, np.float32)
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            k = np.roll(known, (dy, dx), (0, 1)); acc += np.roll(rgb, (dy, dx), (0, 1)) * k[..., None]; cnt += k
        grow = (~known) & (cnt > 0)
        rgb[grow] = acc[grow] / cnt[grow][:, None]; known |= grow
    png = os.path.join(out, f"{name}_albedo.png")
    Image.fromarray((rgb * 255 + 0.5).astype(np.uint8)).save(png)

    nt.nodes.clear()
    o = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled"); bsdf.inputs["Roughness"].default_value = 0.7
    tex = nt.nodes.new("ShaderNodeTexImage"); tex.image = bpy.data.images.load(png); tex.image.pack()
    tex.interpolation = 'Linear'
    nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(bsdf.outputs["BSDF"], o.inputs["Surface"])
    for n in ("pos", "nrm", "id"): bpy.data.images.remove(bpy.data.images[n])
    me.attributes.remove(me.attributes["block"])

    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(out, f"{name}.blend"))
    bpy.ops.export_scene.fbx(
        filepath=os.path.join(out, f"{name}.fbx"), use_selection=True, object_types={'MESH'},
        apply_scale_options='FBX_SCALE_ALL', axis_forward='-Z', axis_up='Y',
        mesh_smooth_type='FACE', use_tspace=False, path_mode='COPY', embed_textures=True, bake_anim=False)
    print(f"{name}: TRIANGLES={len(me.polygons)} dims={tuple(round(d, 1) for d in obj.dimensions)}")

    if RENDER_PREVIEW:
        from mathutils import Euler
        world = bpy.data.worlds.new("W"); scene.world = world
        bg = world.node_tree.nodes["Background"]
        bg.inputs[0].default_value = (0.55, 0.45, 0.25, 1); bg.inputs[1].default_value = 0.8
        sun = bpy.data.objects.new("Sun", bpy.data.lights.new("Sun", 'SUN'))
        sun.data.energy = 2.6; sun.rotation_euler = Euler((math.radians(40), 0, math.radians(-30)))
        scene.collection.objects.link(sun)
        cam = bpy.data.objects.new("Cam", bpy.data.cameras.new("Cam"))
        scene.collection.objects.link(cam); scene.camera = cam
        top = obj.dimensions.z
        target = Vector((0, 0, top * 0.5)); d = Vector((1.0, -1.25, 0.45)).normalized() * top * 2.2
        cam.location = target + d
        cam.rotation_euler = (-d).to_track_quat('-Z', 'Y').to_euler()
        cam.data.lens = 45
        scene.render.resolution_x = scene.render.resolution_y = 700
        scene.cycles.samples = 32; scene.view_settings.view_transform = 'Standard'
        scene.render.filepath = os.path.join(out, "preview.png")
        bpy.ops.render.render(write_still=True)

for name, boxes in TREES.items():
    make_tree(name, boxes)
