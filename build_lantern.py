"""Low-poly stylised stone lantern (toro) for Roblox — built with Blender 5.1.2 (bpy).

Run:  python build_lantern.py [--preview]    (needs the `bpy` 5.1.2 module + Pillow)

Outputs (in ./StoneLantern/):
  StoneLantern.fbx        - Roblox-ready: 'StoneLantern' (stone body, petals, window glow)
                            + 'LanternLight' (inner light cube - set it to Neon in Roblox)
  StoneLantern_albedo.png - baked colour texture (1024x1024) incl. painted stone + glowing kanji
  StoneLantern.blend      - source scene (texture packed)
"""
import bpy, bmesh, math, os, sys, random
from mathutils import Vector, Euler
from mathutils.bvhtree import BVHTree
from PIL import Image, ImageDraw, ImageFont, ImageFilter

NAME = "StoneLantern"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, NAME)
os.makedirs(OUT, exist_ok=True)
RENDER_PREVIEW = "--preview" in sys.argv
FONT = "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf"
TEX = 1024
random.seed(11)

bpy.ops.wm.read_factory_settings(use_empty=True)

# Reference is 900x1600px: ground at y=1450, centre x=450.  1px = 0.006 studs -> ~8.5 studs tall.
S = 0.006
def Z(py): return (1450 - py) * S

STONE, GLOW, PETAL, LIGHT, TEXT = range(5)
bm = bmesh.new()
KIND = bm.faces.layers.int.new("kind")
PART = bm.faces.layers.int.new("part")

def face(vs, kind, part):
    f = bm.faces.new(vs); f[KIND] = kind; f[PART] = part; return f

# ---------------------------------------------------------------- lofted blocks
def ring8(h, c, z):
    """Square with chamfered corners (8 verts, CCW from -Y side's right end)."""
    h *= S; c *= S
    pts = [(h, -h + c), (h, h - c), (h - c, h), (-h + c, h),
           (-h, h - c), (-h, -h + c), (-h + c, -h), (h - c, -h)]
    return [bm.verts.new((x, y, z)) for x, y in pts]

def loft(rings, part, cap_bottom=True, cap_top=True, text_seg=None):
    """rings: [(y_px, half_px, chamfer_px)] bottom -> top."""
    R = [ring8(h, c, Z(y)) for y, h, c in rings]
    for si, (a, b) in enumerate(zip(R, R[1:])):
        n = len(a)
        for j in range(n):
            k = TEXT if (si == text_seg and j == 6) else STONE   # j=6 is the -Y (front) face
            face((a[j], a[(j + 1) % n], b[(j + 1) % n], b[j]), k, part)
    if cap_bottom: face(R[0][::-1], STONE, part)
    if cap_top: face(R[-1], STONE, part)

# base slab
loft([(1450, 225, 45), (1392, 225, 45), (1322, 196, 50), (1292, 168, 45)], part=0)
# pillar: block, waist taper, shaft (inscribed), collar
loft([(1300, 118, 16), (1150, 118, 16), (1108, 82, 12), (830, 80, 12), (800, 94, 13), (790, 94, 13)],
     part=1, cap_bottom=False, text_seg=2)
# middle platform (inverted frustum)
loft([(800, 150, 18), (700, 198, 22), (668, 198, 22), (652, 190, 22)], part=2)

# ---------------------------------------------------------------- fire box with 4 rounded windows
def firebox(W=115, zb_py=652, zt_py=470, t=15, a=80, ob_py=628, ot_py=495, k=20, part=3):
    W, t, a, k = W * S, t * S, a * S, k * S
    zb, zt, ob, ot = Z(zb_py), Z(zt_py), Z(ob_py), Z(ot_py)
    up = Vector((0, 0, 1))
    made = []
    for n in (Vector((1, 0, 0)), Vector((0, 1, 0)), Vector((-1, 0, 0)), Vector((0, -1, 0))):
        r = up.cross(n)
        def at(off, rr, z): return bm.verts.new(n * off + r * rr + up * z)
        # outer and inner wall rectangles (inner corners are pulled in by the wall thickness)
        co = [at(W, -W, zb), at(W, W, zb), at(W, W, zt), at(W, -W, zt)]
        ci = [at(W - t, -(W - t), zb), at(W - t, W - t, zb), at(W - t, W - t, zt), at(W - t, -(W - t), zt)]
        def octo(off):
            return [at(off, -a + k, ob), at(off, a - k, ob), at(off, a, ob + k), at(off, a, ot - k),
                    at(off, a - k, ot), at(off, -a + k, ot), at(off, -a, ot - k), at(off, -a, ob + k)]
        oo, oi = octo(W), octo(W - t)
        for c, o, flip, kind in ((co, oo, False, STONE), (ci, oi, True, GLOW)):
            fs = [(c[0], c[1], o[1], o[0]), (c[1], o[2], o[1]), (c[1], c[2], o[3], o[2]), (c[2], o[4], o[3]),
                  (c[2], c[3], o[5], o[4]), (c[3], o[6], o[5]), (c[3], c[0], o[7], o[6]), (c[0], o[0], o[7])]
            for vs in fs:
                face(vs[::-1] if flip else vs, kind, part)
        for i in range(8):                                  # window reveals
            face((oo[i], oo[(i + 1) % 8], oi[(i + 1) % 8], oi[i]), GLOW, part)
        made += co + ci + oo + oi
    w = W - t
    fl = [bm.verts.new((x * w, y * w, zb + 0.002)) for x, y in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    ce = [bm.verts.new((x * w, y * w, zt)) for x, y in ((-1, -1), (-1, 1), (1, 1), (1, -1))]
    face(fl, GLOW, part); face(ce, GLOW, part)
    bmesh.ops.remove_doubles(bm, verts=made, dist=1e-5)
    # glowing light cube sitting on the floor
    lc = bmesh.ops.create_cube(bm, size=50 * S)
    for v in lc["verts"]:
        v.co += Vector((0, 0, zb + 27 * S))
    for f in {f for v in lc["verts"] for f in v.link_faces}:
        f[KIND] = LIGHT; f[PART] = 7

firebox()

# ---------------------------------------------------------------- pagoda roof with upturned corners
def ring12(h_mid, z_mid, h_corner, z_corner):
    pts = []
    for sx, sy in ((1, -1), (1, 1), (-1, 1), (-1, -1)):             # corners CCW
        pts.append(((sx * h_corner * S, sy * h_corner * S), Z(z_corner)))
        # two edge points on the side leading to the next corner
        nx, ny = {(1, -1): (1, 1), (1, 1): (-1, 1), (-1, 1): (-1, -1), (-1, -1): (1, -1)}[(sx, sy)]
        for f in (0.32, 0.68):
            mx = (sx + (nx - sx) * f) * h_mid * S; my = (sy + (ny - sy) * f) * h_mid * S
            # keep edge points on the straight side
            if sx == nx: mx = sx * h_mid * S
            if sy == ny: my = sy * h_mid * S
            pts.append(((mx, my), Z(z_mid)))
    return [bm.verts.new((x, y, z)) for (x, y), z in pts]

roof = [ring12(140, 470, 140, 470),     # underside, on top of the fire box
        ring12(292, 418, 335, 342),     # eave bottom (corners swept up)
        ring12(292, 384, 338, 306),     # eave top edge
        ring12(222, 330, 238, 292),     # concave slope
        ring12(132, 258, 132, 258)]     # roof top
for a, b in zip(roof, roof[1:]):
    for j in range(12):
        face((a[j], a[(j + 1) % 12], b[(j + 1) % 12], b[j]), STONE, 4)
face(roof[-1], STONE, 4)

# cap block and rounded finial stone
loft([(262, 96, 10), (172, 96, 10), (140, 72, 12)], part=5, cap_bottom=False)
loft([(142, 50, 16), (124, 58, 20), (66, 56, 20), (44, 38, 16)], part=6)

bmesh.ops.recalc_face_normals(bm, faces=[f for f in bm.faces if f[PART] not in (3,)])

# ---------------------------------------------------------------- scattered cherry petals
bm.faces.ensure_lookup_table()
bvh = BVHTree.FromBMesh(bm)

def petal(loc, nrm, size):
    u = nrm.cross(Vector((math.cos(random.uniform(0, 6.3)), math.sin(random.uniform(0, 6.3)), 0.01)))
    u.normalize(); w = nrm.cross(u)
    c = loc + nrm * 0.004
    vs = [c - u * size * 0.5, c + w * size * 0.42, c + u * size * 0.55, c - w * size * 0.42]
    vs = [bm.verts.new(p) for p in vs]
    face(vs, PETAL, 8)

up = Vector((0, 0, 1))
placed = 0
while placed < 26:                                   # on ledges, roof and finial
    x, y = random.uniform(-300, 300) * S, random.uniform(-300, 300) * S
    hit = bvh.ray_cast(Vector((x, y, 12)), Vector((0, 0, -1)))
    if hit[0] is not None and hit[1].z > 0.55 and hit[0].z > 0.05:
        petal(hit[0], hit[1], random.uniform(32, 46) * S); placed += 1
placed = 0
while placed < 40:                                   # on the ground around the base
    ang = random.uniform(0, 2 * math.pi)
    rad = random.uniform(240, 360) * S
    x, y = rad * math.cos(ang), rad * math.sin(ang) * 0.95
    if abs(x) < 232 * S and abs(y) < 232 * S: continue
    petal(Vector((x, y, 0.0)), up, random.uniform(38, 56) * S); placed += 1

bmesh.ops.triangulate(bm, faces=bm.faces[:])

# ---------------------------------------------------------------- colours (stylised light baked in)
L = Vector((-0.5, -0.6, 0.62)).normalized()
def lerp(a, b, t): return tuple(x + (y - x) * t for x, y in zip(a, b))
def clamp(x): return max(0.0, min(1.0, x))
part_z = {}
for f in bm.faces:
    for v in f.verts:
        lo, hi = part_z.get(f[PART], (1e9, -1e9))
        part_z[f[PART]] = (min(lo, v.co.z), max(hi, v.co.z))
ztop = max(hi for lo, hi in part_z.values())

def corner_color(f, v):
    n = f.normal; t = clamp(0.5 + 0.55 * n.dot(L)); k = f[KIND]
    lo, hi = part_z[f[PART]]
    lh = clamp((v.co.z - lo) / max(hi - lo, 1e-6))
    if k in (STONE, TEXT):
        h = clamp(v.co.z / ztop)
        base = lerp((0.13, 0.50, 0.62), (0.27, 0.42, 0.78), clamp(h * 1.6))       # teal base -> blue
        if f[PART] in (5, 6):
            base = (0.47, 0.45, 0.82)                                            # lavender top stones
        g = (0.55 + 0.62 * t) * (0.86 + 0.18 * lh)
        if n.z < -0.4: g *= 0.8
        c = tuple(x * g for x in base); a = 1.0
    elif k == GLOW:
        c = lerp((0.10, 0.62, 0.78), (0.42, 0.98, 0.98), 0.5 + 0.5 * lh); a = 0.0
    elif k == LIGHT:
        c = (0.86, 1.0, 1.0) if n.z > 0.5 else (0.70, 0.98, 1.0); a = 0.0
    else:  # petal
        c = lerp((0.86, 0.66, 0.86), (1.0, 0.86, 0.96), random.random()); a = 0.0
    return tuple(min(1.0, x) for x in c) + (a,)

colors = [[corner_color(f, l.vert) for l in f.loops] for f in bm.faces]

mesh = bpy.data.meshes.new(NAME)
bm.to_mesh(mesh); bm.free()
obj = bpy.data.objects.new(NAME, mesh)
bpy.context.scene.collection.objects.link(obj)
for p in mesh.polygons:
    p.use_smooth = False
col = mesh.color_attributes.new("Col", 'BYTE_COLOR', 'CORNER')
for i, p in enumerate(mesh.polygons):
    for j, li in enumerate(p.loop_indices):
        col.data[li].color_srgb = colors[i][j]
ntri = len(mesh.polygons)

# ---------------------------------------------------------------- UVs: smart project + reserved inscription panel
bpy.context.view_layer.objects.active = obj
obj.select_set(True)
kinds = [a.value for a in mesh.attributes["kind"].data]
uv = mesh.uv_layers.new(name="UVMap")
for p in mesh.polygons:
    p.select = kinds[p.index] != TEXT
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.uv.smart_project(angle_limit=math.radians(55), island_margin=0.006, area_weight=0.0,
                         correct_aspect=True, scale_to_bounds=True)
bpy.ops.object.mode_set(mode='OBJECT')
uv = mesh.uv_layers["UVMap"]          # re-fetch: edit-mode round trip invalidates the old reference
ATLAS = 0.78
TU0, TU1, TV0, TV1 = 0.80, 0.99, 0.60, 0.985          # inscription panel region in UV space
text_pts = [mesh.vertices[v].co for p in mesh.polygons if kinds[p.index] == TEXT for v in p.vertices]
xmin, xmax = min(c.x for c in text_pts), max(c.x for c in text_pts)
zmin, zmax = min(c.z for c in text_pts), max(c.z for c in text_pts)
for p in mesh.polygons:
    for li in p.loop_indices:
        if kinds[p.index] == TEXT:
            co = mesh.vertices[mesh.loops[li].vertex_index].co
            uv.data[li].uv = (TU0 + (co.x - xmin) / (xmax - xmin) * (TU1 - TU0),
                              TV0 + (co.z - zmin) / (zmax - zmin) * (TV1 - TV0))
        else:
            uv.data[li].uv = uv.data[li].uv * ATLAS

# ---------------------------------------------------------------- bake: vertex colour x painted stone noise -> texture
scene = bpy.context.scene
scene.render.engine = 'CYCLES'; scene.cycles.device = 'CPU'; scene.cycles.samples = 8
img = bpy.data.images.new(f"{NAME}_albedo", TEX, TEX, alpha=False)
mat = bpy.data.materials.new(f"{NAME}_Mat")
nt = mat.node_tree; nt.nodes.clear()
N = nt.nodes.new; Lk = nt.links.new
out = N("ShaderNodeOutputMaterial")
attr = N("ShaderNodeVertexColor"); attr.layer_name = "Col"
tc = N("ShaderNodeTexCoord")
noise = N("ShaderNodeTexNoise"); noise.inputs["Scale"].default_value = 3.0; noise.inputs["Detail"].default_value = 4.0
Lk(tc.outputs["Object"], noise.inputs["Vector"])
streak = N("ShaderNodeTexNoise"); streak.inputs["Scale"].default_value = 9.0; streak.inputs["Detail"].default_value = 2.0
Lk(tc.outputs["Object"], streak.inputs["Vector"])
mr = N("ShaderNodeMapRange")                     # mottling 0.86 .. 1.10
mr.inputs["From Min"].default_value = 0.3; mr.inputs["From Max"].default_value = 0.7
mr.inputs["To Min"].default_value = 0.86; mr.inputs["To Max"].default_value = 1.10
Lk(noise.outputs["Fac"], mr.inputs["Value"])
mr2 = N("ShaderNodeMapRange")                    # thin light streaks
mr2.inputs["From Min"].default_value = 0.62; mr2.inputs["From Max"].default_value = 0.72
mr2.inputs["To Min"].default_value = 0.0; mr2.inputs["To Max"].default_value = 0.12
Lk(streak.outputs["Fac"], mr2.inputs["Value"])
add = N("ShaderNodeMath"); add.operation = 'ADD'
Lk(mr.outputs["Result"], add.inputs[0]); Lk(mr2.outputs["Result"], add.inputs[1])
# factor = 1 + mask*(stone - 1)    (mask = vertex alpha, 1 on stone only)
sub = N("ShaderNodeMath"); sub.operation = 'SUBTRACT'; sub.inputs[1].default_value = 1.0
Lk(add.outputs["Value"], sub.inputs[0])
mul = N("ShaderNodeMath"); mul.operation = 'MULTIPLY'
Lk(sub.outputs["Value"], mul.inputs[0]); Lk(attr.outputs["Alpha"], mul.inputs[1])
fac = N("ShaderNodeMath"); fac.operation = 'ADD'; fac.inputs[1].default_value = 1.0
Lk(mul.outputs["Value"], fac.inputs[0])
vm = N("ShaderNodeVectorMath"); vm.operation = 'SCALE'
Lk(attr.outputs["Color"], vm.inputs[0]); Lk(fac.outputs["Value"], vm.inputs["Scale"])
emit = N("ShaderNodeEmission")
Lk(vm.outputs["Vector"], emit.inputs["Color"]); Lk(emit.outputs["Emission"], out.inputs["Surface"])
tex = N("ShaderNodeTexImage"); tex.image = img
nt.nodes.active = tex
obj.data.materials.append(mat)
bpy.ops.object.bake(type='EMIT', margin=4, use_clear=True)
png = os.path.join(OUT, f"{NAME}_albedo.png")
img.filepath_raw = png; img.file_format = 'PNG'; img.save()

# ---------------------------------------------------------------- paint the glowing inscription onto the panel
im = Image.open(png).convert("RGB")
x0, x1 = int(TU0 * TEX), int(TU1 * TEX)
y0, y1 = int((1 - TV1) * TEX), int((1 - TV0) * TEX)
pw, ph = x1 - x0, y1 - y0
layer = Image.new("L", (pw, ph), 0)
d = ImageDraw.Draw(layer)
fs = int(pw * 0.62)
font = ImageFont.truetype(FONT, fs)
glyphs = [("夢", False), ("愛", False), ("漫", True)]       # third glyph is rotated, as on the reference
gap = (ph - 3 * fs) / 4
for i, (ch, flip) in enumerate(glyphs):
    g = Image.new("L", (fs + 8, fs + 8), 0)
    ImageDraw.Draw(g).text((4, 4), ch, font=font, fill=255)
    if flip: g = g.rotate(180)
    layer.paste(g, (int((pw - fs) / 2) - 4, int(gap + i * (fs + gap)) - 4), g)
halo = layer.filter(ImageFilter.GaussianBlur(6))
panel = im.crop((x0, y0, x1, y1))
cyan_halo = Image.new("RGB", panel.size, (60, 230, 230))
panel = Image.composite(cyan_halo, panel, halo.point(lambda v: int(v * 0.55)))
cyan = Image.new("RGB", panel.size, (140, 255, 245))
panel = Image.composite(cyan, panel, layer)
im.paste(panel, (x0, y0))
im.save(png)
img.reload()

# final material: Principled BSDF with the baked texture
for nd in list(nt.nodes):
    if nd not in (out, tex): nt.nodes.remove(nd)
bsdf = N("ShaderNodeBsdfPrincipled")
bsdf.inputs["Roughness"].default_value = 0.85
Lk(tex.outputs["Color"], bsdf.inputs["Base Color"])
Lk(bsdf.outputs["BSDF"], out.inputs["Surface"])
mesh.color_attributes.remove(mesh.color_attributes["Col"])

# ---------------------------------------------------------------- split the light cube into its own part
light_mesh = mesh.copy(); light_mesh.name = "LanternLight"
light = bpy.data.objects.new("LanternLight", light_mesh)
scene.collection.objects.link(light)
for m, keep_light in ((mesh, False), (light_mesh, True)):
    b = bmesh.new(); b.from_mesh(m)
    kl = b.faces.layers.int["kind"]
    bmesh.ops.delete(b, geom=[f for f in b.faces if (f[kl] == LIGHT) != keep_light], context='FACES')
    b.to_mesh(m); b.free()
    for an in ("kind", "part"):
        if an in m.attributes: m.attributes.remove(m.attributes[an])

# ---------------------------------------------------------------- export
img.pack()
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT, f"{NAME}.blend"))
for o in (obj, light): o.select_set(True)
bpy.ops.export_scene.fbx(
    filepath=os.path.join(OUT, f"{NAME}.fbx"),
    use_selection=True, object_types={'MESH'},
    apply_scale_options='FBX_SCALE_ALL', axis_forward='-Z', axis_up='Y',
    mesh_smooth_type='FACE', use_tspace=False,
    path_mode='COPY', embed_textures=True, bake_anim=False)
print(f"TRIANGLES={ntri} (body {len(mesh.polygons)} + light {len(light_mesh.polygons)})  "
      f"dims={tuple(round(d, 2) for d in obj.dimensions)}")

# ---------------------------------------------------------------- optional preview renders
if RENDER_PREVIEW:
    world = bpy.data.worlds.new("W"); scene.world = world
    bg = world.node_tree.nodes["Background"]
    bg.inputs[0].default_value = (0.012, 0.012, 0.02, 1); bg.inputs[1].default_value = 1.0
    sun = bpy.data.objects.new("Sun", bpy.data.lights.new("Sun", 'SUN'))
    sun.data.energy = 3.0; sun.data.color = (0.75, 0.8, 1.0)
    sun.rotation_euler = Euler((math.radians(45), 0, math.radians(-40)))
    scene.collection.objects.link(sun)
    fill = bpy.data.objects.new("Fill", bpy.data.lights.new("Fill", 'SUN'))
    fill.data.energy = 1.2; fill.rotation_euler = Euler((math.radians(70), 0, math.radians(150)))
    scene.collection.objects.link(fill)
    pl = bpy.data.objects.new("Glow", bpy.data.lights.new("Glow", 'POINT'))
    pl.data.energy = 60; pl.data.color = (0.3, 1.0, 1.0); pl.location = (0, 0, Z(560))
    scene.collection.objects.link(pl)
    # preview-only neon look for the light cube (in Roblox: Material = Neon)
    neon = bpy.data.materials.new("PreviewNeon"); nn = neon.node_tree
    nn.nodes["Principled BSDF"].inputs["Emission Color"].default_value = (0.6, 1, 1, 1)
    nn.nodes["Principled BSDF"].inputs["Emission Strength"].default_value = 4.0
    light.data.materials[0] = neon
    cam = bpy.data.objects.new("Cam", bpy.data.cameras.new("Cam"))
    cam.data.type = 'ORTHO'; cam.data.ortho_scale = 9.6
    scene.collection.objects.link(cam); scene.camera = cam
    scene.render.resolution_x, scene.render.resolution_y = 540, 960
    scene.cycles.samples = 48
    scene.view_settings.view_transform = 'Standard'
    for name, ang in [("front", 0), ("threequarter", -35), ("back", 160)]:
        a = math.radians(ang)
        cam.location = Vector((30 * math.sin(a), -30 * math.cos(a), 4.3 + 30 * math.tan(math.radians(12))))
        cam.rotation_euler = Euler((math.radians(78), 0, a))
        scene.render.filepath = os.path.join(OUT, f"preview_{name}.png")
        bpy.ops.render.render(write_still=True)
