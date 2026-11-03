"""Low-poly stylized tree for Roblox — built with Blender 5.1.2 (bpy).

Run:  python build_tree.py            (with the `bpy` 5.1.2 module installed)
  or: blender -b -P build_tree.py

Outputs (next to this script, in ./LowPolyTree/):
  LowPolyTree.fbx        - Roblox-ready mesh, baked texture embedded
  LowPolyTree_albedo.png - baked color texture (1024x1024)
  LowPolyTree.blend      - source scene (texture packed)
"""
import bpy, bmesh, math, os, sys
from mathutils import Vector, Euler
from mathutils.bvhtree import BVHTree

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "LowPolyTree")
os.makedirs(OUT, exist_ok=True)
RENDER_PREVIEW = "--preview" in sys.argv

bpy.ops.wm.read_factory_settings(use_empty=True)

# Reference image is 400x400px; ground at y=365, trunk centred at x=205.
# 1px = 0.035 studs  ->  tree is ~11.4 studs tall.
S = 0.035
def P(px, py, depth=0.0):
    return Vector(((px - 205) * S, depth * S, (365 - py) * S))

LEAF, BARK, DECAL, CORE = 0, 1, 2, 3
bm = bmesh.new()
KIND = bm.faces.layers.int.new("kind")

def face(verts, kind):
    f = bm.faces.new(verts)
    f[KIND] = kind
    return f

# ---------------------------------------------------------------- canopy
def cluster(px, top_py, r_px, depth, tilt_front=0, tilt_side=0, rot=0, n=8):
    """Faceted dome cap + vertical skirt with a jagged (leaf-tip) bottom edge."""
    R = r_px * S
    m = Euler((math.radians(tilt_front), math.radians(tilt_side), math.radians(rot)), 'XYZ').to_matrix()
    o = P(px, top_py, depth)
    V = lambda x, y, z: bm.verts.new(o + m @ Vector((x, y, z)))
    step = 2 * math.pi / n
    apex = V(0, 0, 0)
    ringA = [V(0.58 * R * math.cos((i + .5) * step), 0.58 * R * math.sin((i + .5) * step), -0.26 * R) for i in range(n)]
    ring1 = [V(R * math.cos(i * step), R * math.sin(i * step), -0.50 * R) for i in range(n)]
    jag = []
    for j in range(2 * n):
        a = j * step / 2
        if j % 2 == 0:   # leaf tip, points down
            jag.append(V(0.93 * R * math.cos(a), 0.93 * R * math.sin(a), -1.12 * R))
        else:            # notch between tips
            jag.append(V(0.97 * R * math.cos(a), 0.97 * R * math.sin(a), -0.86 * R))
    bottom = V(0, 0, -0.80 * R)
    for i in range(n):
        k = (i + 1) % n
        face((apex, ringA[(i - 1) % n], ringA[i]), LEAF)            # top fan
        face((ringA[(i - 1) % n], ring1[i], ringA[i]), LEAF)         # dome facets
        face((ringA[i], ring1[i], ring1[k]), LEAF)
        a, b, c = jag[2 * i], jag[2 * i + 1], jag[(2 * i + 2) % (2 * n)]
        face((ring1[i], a, b), LEAF)                                # skirt
        face((ring1[i], b, ring1[k]), LEAF)
        face((ring1[k], b, c), LEAF)
    for j in range(2 * n):
        face((jag[j], bottom, jag[(j + 1) % (2 * n)]), LEAF)         # underside
    return o, m, R

#        x    top   R   depth  front  side  rot  n
CLUSTERS = [
    (195,  40, 100,    0,     0,    0,   0, 8),   # big top crown
    (200,  72,  86,   85,   -16,    0,  20, 8),   # back crown
    (108,  88,  62,   35,     0,  -18,  10, 7),   # upper left
    (292,  74,  52,   40,     0,   16,  25, 7),   # upper right (back)
    (302, 120,  72,  -28,    10,   20,  15, 8),   # right
    (104, 145,  62,  -28,    10,  -20,   5, 7),   # left lower
    (276, 180,  55,   30,     0,   18,  30, 7),   # right lower (back)
    (205, 196,  56,  -58,    18,    0,  12, 7),   # bottom centre (front)
    (185, 168,  70,   85,   -18,   -5,   0, 7),   # back lower
]
FRAMES = [cluster(*c) for c in CLUSTERS]

# dark inner core so no sky shows between clusters
ico = bmesh.ops.create_icosphere(bm, subdivisions=1, radius=1.0)
for v in ico["verts"]:
    v.co = Vector((v.co.x * 118 * S, v.co.y * 95 * S, v.co.z * 80 * S)) + P(200, 175, 20)
for f in {f for v in ico["verts"] for f in v.link_faces}:
    f[KIND] = CORE

# ---------------------------------------------------------------- trunk
def tube(points, radii, n=6, kind=BARK, mods=None, phase=0.0):
    rings = []
    for i, (p, r) in enumerate(zip(points, radii)):
        d = (points[min(i + 1, len(points) - 1)] - points[max(i - 1, 0)]).normalized()
        a = Vector((0, 1, 0)) if abs(d.y) < 0.9 else Vector((1, 0, 0))
        u = d.cross(a).normalized(); w = d.cross(u)
        ring = []
        for j in range(n):
            t = phase + j * 2 * math.pi / n
            k = mods[i][j] if mods and mods[i] else 1.0
            ring.append(bm.verts.new(p + (u * math.cos(t) + w * math.sin(t)) * r * k))
        rings.append(ring)
    for r0, r1 in zip(rings, rings[1:]):
        for j in range(n):
            face((r0[j], r0[(j + 1) % n], r1[(j + 1) % n], r1[j]), kind)
    face(rings[0], kind); face(rings[-1], kind)

# main trunk with flared roots (alternating radius on the base ring)
tube([P(205, 365), P(203, 340), P(202, 306)], [40 * S, 23 * S, 17 * S], n=6,
     mods=[[1.0, 0.72, 1.05, 0.7, 1.0, 0.75], None, None], phase=0.3)
# three branches forking up into the canopy
tube([P(199, 322), P(168, 272), P(150, 225, -10)], [15 * S, 8.5 * S, 6 * S], n=5)
tube([P(203, 322), P(195, 262, -6), P(192, 215, -10)], [16 * S, 9.5 * S, 7 * S], n=5)
tube([P(207, 322), P(236, 268), P(258, 228, 5)], [15 * S, 8.5 * S, 6 * S], n=5)

bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])

# ---------------------------------------------------------------- leaf decals
bm.faces.ensure_lookup_table()
bvh = BVHTree.FromBMesh(bm)

def flower(loc, nrm, size=0.5, spin=45):
    u = nrm.cross(Vector((0, 0, 1)))
    if u.length < 1e-3: u = nrm.cross(Vector((1, 0, 0)))
    u.normalize(); w = nrm.cross(u)
    c = loc + nrm * 0.03
    cv = bm.verts.new(c)
    for k in range(4):
        t = math.radians(spin + 90 * k)
        a = u * math.cos(t) + w * math.sin(t); b = nrm.cross(a)
        face((cv, bm.verts.new(c + a * size * 0.5 - b * size * 0.2),
              bm.verts.new(c + a * size), bm.verts.new(c + a * size * 0.5 + b * size * 0.2)), DECAL)

for px, py, spin in [(176, 108, 40), (78, 212, 20), (290, 222, 35), (305, 135, 10), (205, 225, 30)]:
    hit = bvh.ray_cast(P(px, py, -800), Vector((0, 1, 0)))
    if hit[0] is not None:
        flower(hit[0], hit[1], spin=spin)
for px, py, spin in [(200, 80, 0), (110, 120, 60)]:          # a couple on the back
    hit = bvh.ray_cast(P(px, py, 800), Vector((0, -1, 0)))
    if hit[0] is not None:
        flower(hit[0], hit[1], spin=spin)

def stray_leaf(base, direction, normal, length=0.55, width=0.2):
    d = direction.normalized(); nrm = normal.normalized(); b = nrm.cross(d)
    for side in (1, -1):
        off = nrm * 0.01 * side
        vs = [bm.verts.new(base + off), bm.verts.new(base + d * length * 0.5 + b * width + off),
              bm.verts.new(base + d * length + off), bm.verts.new(base + d * length * 0.5 - b * width + off)]
        face(vs if side == 1 else vs[::-1], DECAL)

def rim_leaf(ci, ang_deg, up=0.6, facing=-1):
    """Leaf sprouting from the dome edge of cluster `ci`, pointing outward."""
    o, m, R = FRAMES[ci]
    a = math.radians(ang_deg)
    radial = m @ Vector((math.cos(a), math.sin(a), 0))
    base = o + m @ Vector((0.85 * R * math.cos(a), 0.85 * R * math.sin(a), -0.45 * R))
    d = (radial + (m @ Vector((0, 0, 1))) * up).normalized()
    view = Vector((0, facing, 0.5))
    nrm = (view - d * view.dot(d)).normalized()   # leaf blade turned toward the viewer side
    stray_leaf(base, d, nrm, length=0.8, width=0.28)

rim_leaf(0, 200, 0.9)    # top-left sprig on the crown
rim_leaf(3, 10, 1.2)     # top-right curl
rim_leaf(4, 0, 0.3)      # right edge
rim_leaf(5, 185, 0.3)    # left edge
rim_leaf(1, 90, 0.8, 1)  # back

bmesh.ops.triangulate(bm, faces=bm.faces[:])

# ---------------------------------------------------------------- colours (stylised light baked in)
L = Vector((-0.45, -0.6, 0.66)).normalized()
def lerp(a, b, t): return tuple(x + (y - x) * t for x, y in zip(a, b))
def clamp(x): return max(0.0, min(1.0, x))
zs = [v.co.z for f in bm.faces if f[KIND] in (LEAF, CORE) for v in f.verts]
zmin, zmax = min(zs), max(zs)

def corner_color(f, v):
    n = f.normal; t = clamp(0.5 + 0.65 * n.dot(L)); k = f[KIND]
    h = clamp((v.co.z - zmin) / (zmax - zmin))
    if k == LEAF:
        if n.z < -0.35:
            c = lerp((0.20, 0.31, 0.04), (0.30, 0.42, 0.06), t)
        else:
            c = lerp((0.36, 0.52, 0.06), (0.86, 0.95, 0.28), t)
        g = 0.78 + 0.27 * h
    elif k == CORE:
        c = lerp((0.17, 0.27, 0.03), (0.30, 0.44, 0.06), t); g = 0.85 + 0.2 * h
    elif k == DECAL:
        c = lerp((0.72, 0.86, 0.18), (0.95, 1.0, 0.42), t); g = 1.0
    else:  # bark
        c = lerp((0.33, 0.17, 0.09), (0.70, 0.43, 0.25), t); g = 0.85 + 0.15 * clamp((v.co.z) / 3.5)
    return tuple(min(1.0, x * g) for x in c) + (1.0,)

colors = [[corner_color(f, l.vert) for l in f.loops] for f in bm.faces]

# ---------------------------------------------------------------- mesh object + UVs (one cell per triangle)
mesh = bpy.data.meshes.new("LowPolyTree")
bm.to_mesh(mesh)
bm.free()
obj = bpy.data.objects.new("LowPolyTree", mesh)
bpy.context.scene.collection.objects.link(obj)
for p in mesh.polygons:
    p.use_smooth = False

ntri = len(mesh.polygons)
G = math.ceil(math.sqrt(ntri)); cell = 1.0 / G
uv = mesh.uv_layers.new(name="UVMap")
col = mesh.color_attributes.new("Col", 'BYTE_COLOR', 'CORNER')
corners = [(0.12, 0.12), (0.88, 0.12), (0.12, 0.88)]
for i, p in enumerate(mesh.polygons):
    cx, cy = (i % G) * cell, (i // G) * cell
    for j, li in enumerate(p.loop_indices):
        uv.data[li].uv = (cx + corners[j][0] * cell, cy + corners[j][1] * cell)
        col.data[li].color_srgb = colors[i][j]

# ---------------------------------------------------------------- bake vertex colours -> texture (Cycles)
scene = bpy.context.scene
scene.render.engine = 'CYCLES'
scene.cycles.device = 'CPU'
scene.cycles.samples = 4
img = bpy.data.images.new("LowPolyTree_albedo", 1024, 1024, alpha=False)
img.colorspace_settings.name = 'sRGB'

mat = bpy.data.materials.new("LowPolyTree_Mat")
mat.use_nodes = True
nt = mat.node_tree; nt.nodes.clear()
out = nt.nodes.new("ShaderNodeOutputMaterial")
attr = nt.nodes.new("ShaderNodeVertexColor"); attr.layer_name = "Col"
emit = nt.nodes.new("ShaderNodeEmission")
nt.links.new(attr.outputs["Color"], emit.inputs["Color"])
nt.links.new(emit.outputs["Emission"], out.inputs["Surface"])
tex = nt.nodes.new("ShaderNodeTexImage"); tex.image = img
nt.nodes.active = tex
obj.data.materials.append(mat)

bpy.context.view_layer.objects.active = obj
obj.select_set(True)
bpy.ops.object.bake(type='EMIT', margin=6, use_clear=True)
png = os.path.join(OUT, "LowPolyTree_albedo.png")
img.filepath_raw = png; img.file_format = 'PNG'; img.save()
img.filepath = png

# final material: plain Principled BSDF with the baked texture (what Roblox reads)
nt.nodes.remove(emit); nt.nodes.remove(attr)
bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
bsdf.inputs["Roughness"].default_value = 0.9
nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
mesh.color_attributes.remove(mesh.color_attributes["Col"])  # colour lives in the texture only

# ---------------------------------------------------------------- export
img.pack()
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT, "LowPolyTree.blend"))
bpy.ops.export_scene.fbx(
    filepath=os.path.join(OUT, "LowPolyTree.fbx"),
    use_selection=True, object_types={'MESH'},
    apply_scale_options='FBX_SCALE_ALL', axis_forward='-Z', axis_up='Y',
    mesh_smooth_type='FACE', use_mesh_modifiers=True, use_tspace=False,
    path_mode='COPY', embed_textures=True, bake_anim=False)
print(f"TRIANGLES={ntri} VERTS={len(mesh.vertices)}  dims={tuple(round(d,2) for d in obj.dimensions)}")

# ---------------------------------------------------------------- optional preview renders
if RENDER_PREVIEW:
    world = bpy.data.worlds.new("W"); scene.world = world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs[0].default_value = (0.42, 0.42, 0.42, 1)
    world.node_tree.nodes["Background"].inputs[1].default_value = 0.9
    sun = bpy.data.objects.new("Sun", bpy.data.lights.new("Sun", 'SUN'))
    sun.data.energy = 3.0; sun.rotation_euler = Euler((math.radians(40), 0, math.radians(-35)))
    scene.collection.objects.link(sun)
    cam = bpy.data.objects.new("Cam", bpy.data.cameras.new("Cam"))
    cam.data.type = 'ORTHO'; cam.data.ortho_scale = 13.5
    scene.collection.objects.link(cam); scene.camera = cam
    scene.render.resolution_x = scene.render.resolution_y = 800
    scene.cycles.samples = 32
    scene.view_settings.view_transform = 'Standard'
    for name, ang in [("front", 0), ("threequarter", 45), ("side", 90), ("back", 180)]:
        a = math.radians(ang)
        cam.location = Vector((30 * math.sin(a), -30 * math.cos(a), 5.6))
        cam.rotation_euler = Euler((math.radians(90), 0, a))
        scene.render.filepath = os.path.join(OUT, f"preview_{name}.png")
        bpy.ops.render.render(write_still=True)
