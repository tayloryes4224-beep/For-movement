"""Low-poly 'stacked cube' tree for Roblox — built with Blender 5.1.2 (bpy).

Run:  python build_cube_tree.py [--preview]   (with the `bpy` 5.1.2 module installed)
  or: blender -b -P build_cube_tree.py -- --preview

Outputs (in ./CubeTree/):
  CubeTree.fbx        - Roblox-ready mesh, baked texture embedded
  CubeTree_albedo.png - baked colour texture (1024x1024)
  CubeTree.blend      - source scene (texture packed)
"""
import bpy, bmesh, math, os, sys
from mathutils import Vector, Euler

NAME = "CubeTree"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, NAME)
os.makedirs(OUT, exist_ok=True)
RENDER_PREVIEW = "--preview" in sys.argv

bpy.ops.wm.read_factory_settings(use_empty=True)

# Reference is 1200x675px; ground at y=660, trunk at x=620.
# 1px = 0.02 studs -> tree is ~12.4 studs tall.
S = 0.02
def P(px, py, depth=0.0):
    return Vector(((px - 620) * S, depth * S, (660 - py) * S))

LEAF, BARK = 0, 1
bm = bmesh.new()
KIND = bm.faces.layers.int.new("kind")
TONE = bm.faces.layers.float.new("tone")   # per-cube brightness variation

# ---------------------------------------------------------------- trunk
def tube(points, radii, n=7):
    rings = []
    for i, (p, r) in enumerate(zip(points, radii)):
        rings.append([bm.verts.new(p + Vector((math.cos(j * 2 * math.pi / n + 0.2),
                                               math.sin(j * 2 * math.pi / n + 0.2), 0)) * r)
                      for j in range(n)])
    faces = []
    for r0, r1 in zip(rings, rings[1:]):
        for j in range(n):
            faces.append(bm.faces.new((r0[j], r0[(j + 1) % n], r1[(j + 1) % n], r1[j])))
    faces += [bm.faces.new(rings[0][::-1]), bm.faces.new(rings[-1])]
    for f in faces:
        f[KIND] = BARK; f[TONE] = 1.0

tube([P(620, 660), P(617, 590), P(614, 470), P(611, 300), P(608, 110)],
     [30 * S, 19 * S, 13 * S, 10 * S, 6 * S])

# ---------------------------------------------------------------- leaf cubes (chamfered boxes)
def chamfer_box(px, py, w, h, d, depth, yaw=0, tilt=0, roll=0, tone=1.0):
    """Box with chamfered edges: 6 faces + 12 edge strips + 8 corner tris = 44 tris."""
    half = (w * S / 2, d * S / 2, h * S / 2)
    c = min(half) * 0.22                        # chamfer size
    m = Euler((math.radians(tilt), math.radians(roll), math.radians(yaw)), 'XYZ').to_matrix()
    o = P(px, py, depth)
    pts = []
    for gx in (-1, 1):
        for gy in (-1, 1):
            for gz in (-1, 1):
                g = (gx, gy, gz)
                for ax in range(3):             # one vertex per face meeting at this corner
                    pts.append(Vector([g[i] * half[i] - (0 if i == ax else g[i] * c) for i in range(3)]))
    verts = [bm.verts.new(o + m @ q) for q in pts]
    hull = bmesh.ops.convex_hull(bm, input=verts)
    for f in hull["geom"]:
        if isinstance(f, bmesh.types.BMFace):
            f[KIND] = LEAF; f[TONE] = tone

#     x    y    w    h    d   depth  yaw tilt roll tone
CUBES = [
    (607,  62, 100,  95,  95,   -5,  12,   4,  -3, 1.06),   # top
    (642,  82,  55,  55,  55,  -55,  -8,   0,   5, 1.08),   # small top front
    (627, 157, 105,  95,  95,  -20, -10,   2,   0, 1.00),
    (565, 195,  88,  80,  85,   15,  15,  -4,  10, 1.04),   # left, tilted
    (668, 255,  95,  95,  95,   20,  20,   0,  -2, 0.96),   # right back
    (550, 300, 105,  95, 100,   10,   8,   0,   4, 1.00),   # left back
    (608, 320, 140, 125, 115,  -60,   4,   0,  -3, 1.00),   # big front
    (692, 372,  80, 105,  95,   30,  25,   0,   0, 0.88),   # right behind (darker)
    (550, 435, 120, 115, 110,  -30,  -8,   0,   6, 1.00),   # bottom left front
    (662, 468, 150, 112, 120,  -15, -20,   0,   0, 0.94),   # bottom right
    (605, 410,  90,  90,  90,   65,   5,   0,   0, 0.92),   # back fill
    (600, 210,  80,  80,  80,   60,  30,   0,   0, 0.95),   # back fill upper
]
for c in CUBES:
    chamfer_box(*c)

bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
bmesh.ops.triangulate(bm, faces=bm.faces[:])

# ---------------------------------------------------------------- colours (stylised light baked in)
L = Vector((-0.4, -0.75, 0.55)).normalized()
def lerp(a, b, t): return tuple(x + (y - x) * t for x, y in zip(a, b))
def clamp(x): return max(0.0, min(1.0, x))
zs = [v.co.z for f in bm.faces if f[KIND] == LEAF for v in f.verts]
zmin, zmax = min(zs), max(zs)

def corner_color(f, v):
    n = f.normal; t = clamp(0.5 + 0.6 * n.dot(L))
    if f[KIND] == LEAF:
        c = lerp((0.17, 0.46, 0.12), (0.54, 0.88, 0.42), t)
        if n.z > 0.6:                                   # tops a touch lighter/yellower
            c = lerp(c, (0.60, 0.90, 0.45), 0.35)
        g = f[TONE] * (0.85 + 0.2 * clamp((v.co.z - zmin) / (zmax - zmin)))
    else:
        c = lerp((0.24, 0.12, 0.07), (0.62, 0.36, 0.22), t)
        g = 0.8 + 0.25 * clamp(v.co.z / 3.0)
    return tuple(min(1.0, x * g) for x in c) + (1.0,)

colors = [[corner_color(f, l.vert) for l in f.loops] for f in bm.faces]

# ---------------------------------------------------------------- mesh object + UVs (one cell per triangle)
mesh = bpy.data.meshes.new(NAME)
bm.to_mesh(mesh); bm.free()
obj = bpy.data.objects.new(NAME, mesh)
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
scene.render.engine = 'CYCLES'; scene.cycles.device = 'CPU'; scene.cycles.samples = 4
img = bpy.data.images.new(f"{NAME}_albedo", 1024, 1024, alpha=False)
img.colorspace_settings.name = 'sRGB'
mat = bpy.data.materials.new(f"{NAME}_Mat")
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
png = os.path.join(OUT, f"{NAME}_albedo.png")
img.filepath_raw = png; img.file_format = 'PNG'; img.save(); img.filepath = png

nt.nodes.remove(emit); nt.nodes.remove(attr)
bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
bsdf.inputs["Roughness"].default_value = 0.9
nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
mesh.color_attributes.remove(mesh.color_attributes["Col"])

# ---------------------------------------------------------------- export
img.pack()
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT, f"{NAME}.blend"))
bpy.ops.export_scene.fbx(
    filepath=os.path.join(OUT, f"{NAME}.fbx"),
    use_selection=True, object_types={'MESH'},
    apply_scale_options='FBX_SCALE_ALL', axis_forward='-Z', axis_up='Y',
    mesh_smooth_type='FACE', use_tspace=False,
    path_mode='COPY', embed_textures=True, bake_anim=False)
print(f"TRIANGLES={ntri} VERTS={len(mesh.vertices)}  dims={tuple(round(d,2) for d in obj.dimensions)}")

# ---------------------------------------------------------------- optional preview renders
if RENDER_PREVIEW:
    world = bpy.data.worlds.new("W"); scene.world = world
    bg = world.node_tree.nodes["Background"]
    bg.inputs[0].default_value = (0.05, 0.05, 0.05, 1); bg.inputs[1].default_value = 1.0
    sun = bpy.data.objects.new("Sun", bpy.data.lights.new("Sun", 'SUN'))
    sun.data.energy = 3.0; sun.rotation_euler = Euler((math.radians(40), 0, math.radians(-30)))
    scene.collection.objects.link(sun)
    cam = bpy.data.objects.new("Cam", bpy.data.cameras.new("Cam"))
    cam.data.type = 'ORTHO'; cam.data.ortho_scale = 13.5
    scene.collection.objects.link(cam); scene.camera = cam
    scene.render.resolution_x, scene.render.resolution_y = 600, 800
    scene.cycles.samples = 32
    scene.view_settings.view_transform = 'Standard'
    for name, ang in [("front", 0), ("threequarter", 45), ("side", 90), ("back", 180)]:
        a = math.radians(ang)
        cam.location = Vector((30 * math.sin(a), -30 * math.cos(a), 6.2))
        cam.rotation_euler = Euler((math.radians(90), 0, a))
        scene.render.filepath = os.path.join(OUT, f"preview_{name}.png")
        bpy.ops.render.render(write_still=True)
