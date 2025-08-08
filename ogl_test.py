import OpenGL.GL as gl
import OpenGL.GLUT as glut

def check_opengl():
    try:
        glut.glutInit()
        print("✅ OpenGL is available.")
        print("OpenGL version:", gl.glGetString(gl.GL_VERSION).decode())
        print("Renderer:", gl.glGetString(gl.GL_RENDERER).decode())
        print("Vendor:", gl.glGetString(gl.GL_VENDOR).decode())
    except Exception as e:
        print("❌ OpenGL not available:", e)

check_opengl()
