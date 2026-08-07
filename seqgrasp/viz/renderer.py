import mujoco
def render_frame(model,data,width=640,height=480):
    with mujoco.Renderer(model,height,width) as renderer:
        renderer.update_scene(data); return renderer.render()

