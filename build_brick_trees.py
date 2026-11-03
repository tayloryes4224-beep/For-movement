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
WOOD_COL = (0.47, 0.27, 0.15)
LEAF_COL = (0.44, 0.76, 0.27)

def W(x0, x1, y0, y1, z0, z1): return (WOOD, x0, x1, y0, y1, z0, z1)
def G(x0, x1, y0, y1, z0, z1): return (LEAF, x0, x1, y0, y1, z0, z1)

TREES = {
    # big tree from the reference: thick square trunk, chunky brown branch blocks carrying leaf cubes,
    # and a stepped crown of leaf cubes with a gap showing the trunk top
    "BrickOak": [
        W(-3, 3, -3, 3, 0, 31),
        # tier 1: four branch blocks, a leaf cube on each tip
        W(3, 11, -2, 2, 12, 15), W(-11, -3, -1, 3, 13, 16), W(-2, 2, 3, 10, 11, 14), W(-1, 3, -10, -3, 12, 15),
        G(7, 14, -4, 3, 15, 20), G(-14, -7, -2, 5, 16, 21), G(-4, 3, 7, 14, 14, 19), G(-3, 4, -14, -7, 15, 20),
        # tier 2: diagonal-ish branches + leaf cubes
        W(3, 9, 2, 5, 20, 23), W(-9, -3, -5, -2, 21, 24), W(-5, -2, 3, 9, 20, 23), W(2, 5, -9, -3, 21, 24),
        G(6, 13, 2, 9, 23, 28), G(-13, -6, -9, -2, 24, 29), G(-9, -2, 6, 13, 23, 28), G(2, 9, -13, -6, 24, 29),
        # crown
        G(-9, 0, -8, 1, 29, 34), G(0, 9, -1, 8, 29, 34), G(-1, 8, -9, -2, 30, 35), G(-8, 0, 2, 9, 30, 35),
        G(-6, 4, -5, 5, 34, 39), G(-2, 6, -6, 2, 35, 39), G(-3, 3, -3, 3, 39, 43),
    ],
    # the small stacked tree on the left of the reference: brown block under stacked leaf cubes
    "BrickPine": [
        W(-2, 2, -2, 2, 0, 8), W(-4, 4, -4, 4, 6, 10),
        G(-6, 6, -6, 6, 10, 15), G(4, 9, -3, 3, 11, 15),
        G(-5, 4, -4, 5, 15, 19), G(-3, 3, -3, 3, 19, 23), G(-2, 2, -1, 3, 23, 26),
    ],
    # a smaller oak in the same style
    "BrickOakSmall": [
        W(-2, 2, -2, 2, 0, 15),
        W(2, 8, -1, 2, 9, 12), W(-8, -2, -2, 1, 10, 13),
        G(5, 11, -3, 4, 12, 17), G(-11, -5, -4, 3, 13, 18),
        G(-6, 6, -6, 6, 15, 20), G(-4, 1, 3, 9, 16, 20),
        G(-4, 5, -5, 4, 20, 24), G(-2, 3, -2, 3, 24, 27),
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
    """Classic Roblox surfaces: raised square 'Studs' on leaves, recessed 'Inlet' squares on wood.
    One stud per 1x1 grid cell, bevel lit from the top-left of each face; no outlines - block edges
    read only through the per-face shading, like the reference."""
    rng = random.Random(seed)
    tone = np.array([rng.uniform(0.97, 1.03) for _ in boxes] + [1.0], np.float32)
    kind = np.array([b[0] for b in boxes] + [0])
    ID = np.clip(ID, 0, len(boxes))
    x, y, z = P[..., 0], P[..., 1], P[..., 2]
    nx, ny, nz = N[..., 0], N[..., 1], N[..., 2]
    ax = np.argmax(np.abs(N), axis=-1)
    # in-plane coords: u to the viewer's right, v up (for top/bottom faces v = +y)
    u = np.where(ax == 0, -np.sign(nx) * y, np.where(ax == 1, np.sign(ny) * x, x))
    v = np.where(ax == 2, y, z)
    leaf = kind[ID] == LEAF
    base = np.where(leaf[..., None], np.array(LEAF_COL), np.array(WOOD_COL)) * tone[ID][..., None]
    # flat per-face lighting (top bright, sides by direction, bottom dark)
    Lh = np.array((-0.6, -0.8)); side = np.clip(nx * Lh[0] + ny * Lh[1], 0, 1)
    shade = np.where(nz > 0.5, 1.10, np.where(nz < -0.5, 0.70, 0.80 + 0.16 * side))
    # rounded-square stud in every cell
    fu = u - np.floor(u) - 0.5; fv = v - np.floor(v) - 0.5
    h, r, bev = 0.25, 0.07, 0.065
    qx = np.abs(fu) - (h - r); qy = np.abs(fv) - (h - r)
    d = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0) - r
    vert = np.abs(fv) > np.abs(fu)                      # nearest edge is top/bottom
    lit_edge = np.where(vert, fv > 0, fu < 0)           # top or left edge
    band = (d < 0) & (d > -bev)
    f = np.ones_like(d)
    raised = np.where(band, np.where(lit_edge, 1.24, 0.74), 1.0)
    raised = np.where((d >= 0) & (d < 0.05) & ~lit_edge, 0.9, raised)          # small cast shadow
    inlet = np.where(band, np.where(lit_edge, 0.72, 1.14), np.where(d < 0, 0.9, 1.0))
    f = np.where(leaf, raised, inlet)
    return np.clip(base * (shade * f)[..., None], 0, 1)

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
