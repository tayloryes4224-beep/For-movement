import bpy, sys, math
from mathutils import Vector, Euler
name = sys.argv[sys.argv.index("--") + 1]
bpy.ops.wm.open_mainfile(filepath=f"BrickTrees/{name}/{name}.blend")
sc = bpy.context.scene
sc.render.engine = 'CYCLES'; sc.cycles.device = 'CPU'; sc.cycles.samples = 24
w = bpy.data.worlds.new("W"); sc.world = w
bg = w.node_tree.nodes["Background"]; bg.inputs[0].default_value = (0.6, 0.45, 0.2, 1); bg.inputs[1].default_value = 1.0
sun = bpy.data.objects.new("Sun", bpy.data.lights.new("Sun", 'SUN')); sun.data.energy = 1.2
sun.rotation_euler = Euler((math.radians(30), 0, math.radians(-40))); sc.collection.objects.link(sun)
obj = bpy.data.objects[name]; H = obj.dimensions.z
cam = bpy.data.objects.new("Cam", bpy.data.cameras.new("Cam")); sc.collection.objects.link(cam); sc.camera = cam
cam.data.lens = 24
cam.location = Vector((H * 0.25, -H * 0.55, H * 0.12))
tgt = Vector((0, 0, H * 0.62))
cam.rotation_euler = (tgt - cam.location).to_track_quat('-Z', 'Y').to_euler()
sc.render.resolution_x, sc.render.resolution_y = 1000, 700
sc.view_settings.view_transform = 'Standard'
sc.render.filepath = f"BrickTrees/{name}/preview_lowangle.png"
bpy.ops.render.render(write_still=True)
