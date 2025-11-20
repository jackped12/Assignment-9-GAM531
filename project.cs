using OpenTK.Graphics.OpenGL4;
using OpenTK.Mathematics;
using OpenTK.Windowing.Common;
using OpenTK.Windowing.Desktop;
using OpenTK.Windowing.GraphicsLibraryFramework;
using System;
using System.Collections.Generic;

namespace Windows_Engine
{
    public interface ICollider
    {
        bool Intersects(ICollider other);
    }
    public class AABB3D : ICollider
    {
        public Vector3 Min, Max;
        public AABB3D(Vector3 min, Vector3 max) { Min = min; Max = max; }
        public bool Intersects(ICollider other)
        {
            if (other is AABB3D box)
                return (Min.X <= box.Max.X && Max.X >= box.Min.X) &&
                       (Min.Y <= box.Max.Y && Max.Y >= box.Min.Y) &&
                       (Min.Z <= box.Max.Z && Max.Z >= box.Min.Z);
            if (other is SphereCollider sphere)
            {
                Vector3 clamped = Vector3.Clamp(sphere.Center, Min, Max);
                return (clamped - sphere.Center).LengthSquared <= sphere.Radius * sphere.Radius;
            }
            return false;
        }
    }
    public class SphereCollider : ICollider
    {
        public Vector3 Center;
        public float Radius;
        public SphereCollider(Vector3 center, float radius) { Center = center; Radius = radius; }
        public bool Intersects(ICollider other)
        {
            if (other is SphereCollider sphere)
                return (Center - sphere.Center).LengthSquared <= (Radius + sphere.Radius) * (Radius + sphere.Radius);
            if (other is AABB3D box)
                return box.Intersects(this);
            return false;
        }
    }
    public class GameObject3D
    {
        public Vector3 Position;
        public Vector3 Scale = Vector3.One;
        public bool IsActive = true;
        public ICollider? Collider;
    }
    public class Ball : GameObject3D
    {
        public Vector3 Velocity;
        public bool IsActive = true;
        public Vector3 Color = new Vector3(1, 1, 1); // default white
        public Action<Ball>? OnHitWall;
    }
    public class Camera
    {
        public Vector3 Position = new Vector3(0f, 1.5f, 3f);
        private Vector3 _front = new Vector3(0f, 0f, -1f);
        public Vector3 Front => _front;
        public Vector3 Up = Vector3.UnitY;
        public float Yaw { get; private set; } = -MathHelper.PiOver2;
        public float Pitch { get; private set; }
        public float MouseSensitivity { get; set; } = 0.2f;
        public Matrix4 GetViewMatrix() => Matrix4.LookAt(Position, Position + _front, Up);
        public void UpdateDirection(float xoffset, float yoffset)
        {
            Yaw -= xoffset * MouseSensitivity;
            Pitch += yoffset * MouseSensitivity;
            Pitch = MathHelper.Clamp(Pitch, -MathHelper.PiOver2 + 0.01f, MathHelper.PiOver2 - 0.01f);
            _front.X = (float)(Math.Cos(Pitch) * Math.Sin(Yaw));
            _front.Y = (float)(Math.Sin(Pitch));
            _front.Z = (float)(Math.Cos(Pitch) * Math.Cos(Yaw));
            _front = Vector3.Normalize(_front);
        }
    }
    public static class ShapeFactory
    {
        public static float[] CreateCubeTextured()
        {
            float s = 0.5f;
            return new float[]
            {
                // back face
                -s,-s,-s, 0,0,-1, 0,0,
                 s,-s,-s, 0,0,-1, 1,0,
                 s, s,-s, 0,0,-1, 1,1,
                 s, s,-s, 0,0,-1, 1,1,
                -s, s,-s, 0,0,-1, 0,1,
                -s,-s,-s, 0,0,-1, 0,0,
                // front face
                -s,-s, s, 0,0,1, 0,0,
                 s,-s, s, 0,0,1, 1,0,
                 s, s, s, 0,0,1, 1,1,
                 s, s, s, 0,0,1, 1,1,
                -s, s, s, 0,0,1, 0,1,
                -s,-s, s, 0,0,1, 0,0,
                // left face
                -s, s, s, -1,0,0, 1,0,
                -s, s,-s, -1,0,0, 1,1,
                -s, -s,-s, -1,0,0, 0,1,
                -s, -s,-s, -1,0,0, 0,1,
                -s, -s, s, -1,0,0, 0,0,
                -s, s, s, -1,0,0, 1,0,
                // right face
                 s, s, s, 1,0,0, 1,0,
                 s, s,-s, 1,0,0, 1,1,
                 s, -s,-s, 1,0,0, 0,1,
                 s, -s,-s, 1,0,0, 0,1,
                 s, -s, s, 1,0,0, 0,0,
                 s, s, s, 1,0,0, 1,0,
                // bottom face
                -s, -s,-s, 0,-1,0, 0,1,
                 s, -s,-s, 0,-1,0, 1,1,
                 s, -s, s, 0,-1,0, 1,0,
                 s, -s, s, 0,-1,0, 1,0,
                -s, -s, s, 0,-1,0, 0,0,
                -s, -s,-s, 0,-1,0, 0,1,
                // top face
                -s, s,-s, 0,1,0, 0,1,
                 s, s,-s, 0,1,0, 1,1,
                 s, s, s, 0,1,0, 1,0,
                 s, s, s, 0,1,0, 1,0,
                -s, s, s, 0,1,0, 0,0,
                -s, s,-s, 0,1,0, 0,1,
            };
        }
        public static float[] CreatePlaneTextured(float size)
        {
            float s = size / 2;
            return new float[]
            {
                -s, 0, -s, 0,1,0, 0,0,
                 s, 0, -s, 0,1,0, 1,0,
                 s, 0, s, 0,1,0, 1,1,
                 s, 0, s, 0,1,0, 1,1,
                -s, 0, s, 0,1,0, 0,1,
                -s, 0, -s, 0,1,0, 0,0,
            };
        }
    }
    public static class TextureLoader
    {
        public static int LoadTexture()
        {
            int tex = GL.GenTexture();
            GL.BindTexture(TextureTarget.Texture2D, tex);
            byte[] whitePixel = { 255, 255, 255, 255 };
            GL.TexImage2D(TextureTarget.Texture2D, 0, PixelInternalFormat.Rgba, 1, 1, 0,
                PixelFormat.Rgba, PixelType.UnsignedByte, whitePixel);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.Nearest);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Nearest);
            GL.GenerateMipmap(GenerateMipmapTarget.Texture2D);
            return tex;
        }
    }

    public class Game : GameWindow
    {
        private int shaderProgram;
        private int vaoCube, vaoGround;
        private int texGround, texCube;

        private Camera camera = new();

        private List<GameObject3D> walls = new();
        private GameObject3D door;

        private GameObject3D playerObject;
        private List<Ball> balls = new();

        private bool firstMouse = true;
        private Vector2 lastMousePos;
        private float fov = 60f;
        private Matrix4 projection;

        private bool isThrowReady = true;
        private float throwCooldown = 0f;

        public Game(GameWindowSettings gws, NativeWindowSettings nws) : base(gws, nws)
        {
            CursorState = CursorState.Grabbed;
        }
        protected override void OnLoad()
        {
            base.OnLoad();
            GL.ClearColor(0.1f, 0.12f, 0.15f, 1f);
            GL.Enable(EnableCap.DepthTest);
            SetupShaders();

            // Setup multiple walls (breakable)
            walls.Add(new GameObject3D()
            {
                Position = new Vector3(-3, 0.5f, -4),
                Scale = new Vector3(1, 1, 0.2f),
                Collider = new AABB3D(new Vector3(-3.5f, 0, -4.1f), new Vector3(-2.5f, 1, -3.9f)),
                IsActive = true
            });
            walls.Add(new GameObject3D()
            {
                Position = new Vector3(0, 0.5f, -4),
                Scale = new Vector3(1, 1, 0.2f),
                Collider = new AABB3D(new Vector3(-0.5f, 0, -4.1f), new Vector3(0.5f, 1, -3.9f)),
                IsActive = true
            });
            walls.Add(new GameObject3D()
            {
                Position = new Vector3(3, 0.5f, -4),
                Scale = new Vector3(1, 1, 0.2f),
                Collider = new AABB3D(new Vector3(2.5f, 0, -4.1f), new Vector3(3.5f, 1, -3.9f)),
                IsActive = true
            });

            // Door object (not breakable)
            door = new GameObject3D()
            {
                Position = new Vector3(0, 0.5f, -2),
                Scale = new Vector3(1.5f, 1.5f, 0.2f),
                Collider = new AABB3D(new Vector3(-0.75f, 0, -2.1f), new Vector3(0.75f, 1.5f, -1.9f)),
                IsActive = true
            };

            playerObject = new GameObject3D()
            {
                Position = new Vector3(0f, 0f, 3f),
                Scale = new Vector3(0.3f, 1.8f, 0.3f),
                Collider = new AABB3D(new Vector3(-0.15f, 0, 2.85f), new Vector3(0.15f, 1.8f, 3.15f)),
                IsActive = true
            };

            vaoGround = CreateMesh(ShapeFactory.CreatePlaneTextured(10f));
            vaoCube = CreateMesh(ShapeFactory.CreateCubeTextured());
            texCube = TextureLoader.LoadTexture();
            texGround = TextureLoader.LoadTexture();

            projection = Matrix4.CreatePerspectiveFieldOfView(
                MathHelper.DegreesToRadians(fov), Size.X / (float)Size.Y, 0.1f, 100f);
        }

        private void SetupShaders()
        {
            int v = GL.CreateShader(ShaderType.VertexShader);
            GL.ShaderSource(v, VertexShaderSource);
            GL.CompileShader(v);
            CheckShaderCompile(v);

            int f = GL.CreateShader(ShaderType.FragmentShader);
            GL.ShaderSource(f, FragmentShaderSource);
            GL.CompileShader(f);
            CheckShaderCompile(f);

            shaderProgram = GL.CreateProgram();
            GL.AttachShader(shaderProgram, v);
            GL.AttachShader(shaderProgram, f);
            GL.LinkProgram(shaderProgram);
            CheckProgramLink(shaderProgram);

            GL.DeleteShader(v);
            GL.DeleteShader(f);
        }

        protected override void OnUpdateFrame(FrameEventArgs e)
        {
            base.OnUpdateFrame(e);
            var kb = KeyboardState;
            var ms = MouseState;
            float dt = (float)e.Time;

            if (kb.IsKeyDown(Keys.Escape))
                Close();

            if (firstMouse)
            {
                lastMousePos = ms.Position;
                firstMouse = false;
            }
            else
            {
                float deltaX = ms.Position.X - lastMousePos.X;
                float deltaY = lastMousePos.Y - ms.Position.Y;
                lastMousePos = ms.Position;
                camera.UpdateDirection(deltaX * camera.MouseSensitivity, deltaY * camera.MouseSensitivity);
            }

            Vector3 dir = Vector3.Zero;
            Vector3 forward = camera.Front;
            Vector3 right = Vector3.Normalize(Vector3.Cross(forward, Vector3.UnitY));
            if (kb.IsKeyDown(Keys.W)) dir += forward;
            if (kb.IsKeyDown(Keys.S)) dir -= forward;
            if (kb.IsKeyDown(Keys.A)) dir -= right;
            if (kb.IsKeyDown(Keys.D)) dir += right;
            if (dir.LengthSquared > 0) dir = dir.Normalized();

            playerObject.Position += dir * 3f * dt;
            if (playerObject.Collider is AABB3D box)
            {
                box.Min = playerObject.Position - playerObject.Scale / 2;
                box.Max = playerObject.Position + playerObject.Scale / 2;
            }

            // Throw ball
            if (kb.IsKeyDown(Keys.Space) && isThrowReady)
            {
                Vector3 spawnPos = playerObject.Position + camera.Front * 0.5f + new Vector3(0, 1f, 0);
                var ball = new Ball
                {
                    Position = spawnPos,
                    Velocity = camera.Front * 10f,
                    Collider = new SphereCollider(spawnPos, 0.2f),
                    IsActive = true,
                    Color = new Vector3(1f, 1f, 1f) // white initially
                };
                balls.Add(ball);
                isThrowReady = false;
                throwCooldown = 0.3f;
            }

            if (!isThrowReady)
            {
                throwCooldown -= dt;
                if (throwCooldown <= 0f)
                    isThrowReady = true;
            }

            // Update balls and check collisions
            foreach (var ball in balls)
            {
                if (!ball.IsActive) continue;
                ball.Position += ball.Velocity * dt;
                if (ball.Collider is SphereCollider sph) sph.Center = ball.Position;

                // Check collision with walls (breakable)
                foreach (var wall in walls)
                {
                    if (!wall.IsActive || wall.Collider == null) continue;
                    if (ball.Collider != null && ball.Collider.Intersects(wall.Collider))
                    {
                        wall.IsActive = false;
                        ball.IsActive = false;
                        break;
                    }
                }
                if (!ball.IsActive) continue;

                // Check collision with door (reflect ball)
                if (door.IsActive && door.Collider != null && ball.Collider != null && ball.Collider.Intersects(door.Collider))
                {
                    // Reflect velocity on door hit
                    ball.Velocity = -ball.Velocity;
                    // Change ball color to red
                    ball.Color = new Vector3(1, 0, 0);
                }
            }
            balls.RemoveAll(b => !b.IsActive);
        }

        protected override void OnRenderFrame(FrameEventArgs e)
        {
            base.OnRenderFrame(e);
            GL.Clear(ClearBufferMask.ColorBufferBit | ClearBufferMask.DepthBufferBit);
            GL.UseProgram(shaderProgram);

            GL.Uniform3(GL.GetUniformLocation(shaderProgram, "viewPos"), ref camera.Position);
            GL.Uniform3(GL.GetUniformLocation(shaderProgram, "lightPos"), new Vector3(0, 4, 4));
            GL.Uniform3(GL.GetUniformLocation(shaderProgram, "lightColor"), new Vector3(1, 1, 1));
            GL.Uniform1(GL.GetUniformLocation(shaderProgram, "lightIntensity"), 2.0f);
            GL.Uniform3(GL.GetUniformLocation(shaderProgram, "matAmbient"), new Vector3(0.2f));
            GL.Uniform3(GL.GetUniformLocation(shaderProgram, "matSpecular"), new Vector3(0.8f));
            GL.Uniform1(GL.GetUniformLocation(shaderProgram, "matShininess"), 64f);

            var view = camera.GetViewMatrix();
            GL.UniformMatrix4(GL.GetUniformLocation(shaderProgram, "view"), false, ref view);
            GL.UniformMatrix4(GL.GetUniformLocation(shaderProgram, "projection"), false, ref projection);

            // Draw ground
            GL.BindVertexArray(vaoGround);
            GL.BindTexture(TextureTarget.Texture2D, texGround);
            GL.Uniform1(GL.GetUniformLocation(shaderProgram, "uTexture"), 0);
            var groundModel = Matrix4.CreateTranslation(0, -0.01f, 0);
            GL.UniformMatrix4(GL.GetUniformLocation(shaderProgram, "model"), false, ref groundModel);
            GL.DrawArrays(PrimitiveType.Triangles, 0, 6);

            // Draw walls (breakable, gray)
            GL.BindVertexArray(vaoCube);
            GL.BindTexture(TextureTarget.Texture2D, texCube);
            GL.Uniform1(GL.GetUniformLocation(shaderProgram, "uTexture"), 0);
            GL.Uniform3(GL.GetUniformLocation(shaderProgram, "matDiffuse"), new Vector3(0.4f, 0.4f, 0.4f));
            foreach (var w in walls)
            {
                if (!w.IsActive) continue;
                var wallModel = Matrix4.CreateScale(w.Scale) * Matrix4.CreateTranslation(w.Position);
                GL.UniformMatrix4(GL.GetUniformLocation(shaderProgram, "model"), false, ref wallModel);
                GL.DrawArrays(PrimitiveType.Triangles, 0, 36);
            }

            // Draw door in green
            GL.Uniform3(GL.GetUniformLocation(shaderProgram, "matDiffuse"), new Vector3(0f, 0.7f, 0.2f));
            var doorModel = Matrix4.CreateScale(door.Scale) * Matrix4.CreateTranslation(door.Position);
            GL.UniformMatrix4(GL.GetUniformLocation(shaderProgram, "model"), false, ref doorModel);
            GL.DrawArrays(PrimitiveType.Triangles, 0, 36);

            // Draw player (blue)
            GL.Uniform3(GL.GetUniformLocation(shaderProgram, "matDiffuse"), new Vector3(0.2f, 0.4f, 0.7f));
            var playerModel = Matrix4.CreateScale(playerObject.Scale) * Matrix4.CreateTranslation(playerObject.Position);
            GL.UniformMatrix4(GL.GetUniformLocation(shaderProgram, "model"), false, ref playerModel);
            GL.DrawArrays(PrimitiveType.Triangles, 0, 36);

            // Draw balls with their own color
            foreach (var ball in balls)
            {
                if (!ball.IsActive) continue;
                GL.Uniform3(GL.GetUniformLocation(shaderProgram, "matDiffuse"), ball.Color);
                var ballModel = Matrix4.CreateScale(0.2f) * Matrix4.CreateTranslation(ball.Position);
                GL.UniformMatrix4(GL.GetUniformLocation(shaderProgram, "model"), false, ref ballModel);
                GL.DrawArrays(PrimitiveType.Triangles, 0, 36);
            }

            SwapBuffers();
        }

        int CreateMesh(float[] vertices)
        {
            int vao = GL.GenVertexArray();
            int vbo = GL.GenBuffer();
            GL.BindVertexArray(vao);
            GL.BindBuffer(BufferTarget.ArrayBuffer, vbo);
            GL.BufferData(BufferTarget.ArrayBuffer, vertices.Length * sizeof(float), vertices, BufferUsageHint.StaticDraw);
            int stride = 8 * sizeof(float);
            GL.VertexAttribPointer(0, 3, VertexAttribPointerType.Float, false, stride, 0);
            GL.EnableVertexAttribArray(0);
            GL.VertexAttribPointer(1, 3, VertexAttribPointerType.Float, false, stride, 3 * sizeof(float));
            GL.EnableVertexAttribArray(1);
            GL.VertexAttribPointer(2, 2, VertexAttribPointerType.Float, false, stride, 6 * sizeof(float));
            GL.EnableVertexAttribArray(2);
            GL.BindVertexArray(0);
            return vao;
        }

        void CheckShaderCompile(int shader)
        {
            GL.GetShader(shader, ShaderParameter.CompileStatus, out int status);
            if (status == (int)All.False)
                throw new Exception(GL.GetShaderInfoLog(shader));
        }
        void CheckProgramLink(int program)
        {
            GL.GetProgram(program, GetProgramParameterName.LinkStatus, out int status);
            if (status == (int)All.False)
                throw new Exception(GL.GetProgramInfoLog(program));
        }

        protected override void OnResize(ResizeEventArgs e)
        {
            base.OnResize(e);
            GL.Viewport(0, 0, Size.X, Size.Y);
            float aspect = Size.X / (float)Size.Y;
            projection = Matrix4.CreatePerspectiveFieldOfView(MathHelper.DegreesToRadians(fov), aspect, 0.1f, 100f);
        }

        public static string VertexShaderSource = @"
            #version 330 core
            layout (location = 0) in vec3 aPos;
            layout (location = 1) in vec3 aNormal;
            layout (location = 2) in vec2 aTexCoord;
            out vec3 FragPos;
            out vec3 Normal;
            out vec2 TexCoord;
            uniform mat4 model;
            uniform mat4 view;
            uniform mat4 projection;
            void main()
            {
                FragPos = vec3(model * vec4(aPos, 1.0));
                Normal = mat3(transpose(inverse(model))) * aNormal;
                TexCoord = aTexCoord;
                gl_Position = projection * view * vec4(FragPos, 1.0);
            }
        ";
        public static string FragmentShaderSource = @"
            #version 330 core
            out vec4 FragColor;
            in vec3 FragPos;
            in vec3 Normal;
            in vec2 TexCoord;
            uniform vec3 lightPos;
            uniform vec3 viewPos;
            uniform vec3 lightColor;
            uniform float lightIntensity;
            uniform vec3 matAmbient;
            uniform vec3 matDiffuse;
            uniform vec3 matSpecular;
            uniform float matShininess;
            uniform sampler2D uTexture;
            void main()
            {
                vec3 texColor = texture(uTexture, TexCoord).rgb;
                vec3 ambient = matAmbient * lightColor * lightIntensity;
                vec3 norm = normalize(Normal);
                vec3 lightDir = normalize(lightPos - FragPos);
                float diff = max(dot(norm, lightDir), 0.0);
                vec3 diffuse = diff * matDiffuse * lightColor * lightIntensity;
                vec3 viewDir = normalize(viewPos - FragPos);
                vec3 reflectDir = reflect(-lightDir, norm);
                float spec = pow(max(dot(viewDir, reflectDir), 0.0), matShininess);
                vec3 specular = spec * matSpecular * lightColor * lightIntensity;
                vec3 result = (ambient + diffuse + specular) * texColor;
                FragColor = vec4(result, 1.0);
            }
        ";
    }
    public static class Program
    {
        public static void Main()
        {
            var nativeWindowSettings = new NativeWindowSettings()
            {
                Size = new Vector2i(800, 600),
                Title = "OpenTK Multiple Walls and Reflective Door"
            };
            using var window = new Game(GameWindowSettings.Default, nativeWindowSettings);
            window.Run();
        }
    }
}
