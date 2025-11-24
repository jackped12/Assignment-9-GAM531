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
                // Find the closest point on the AABB to the sphere center.
                Vector3 clamped = Vector3.Clamp(sphere.Center, Min, Max);

                // Check if the distance squared from the closest point to the center is less than the radius squared.
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
        public Vector3 Rotation = Vector3.Zero; // In radians
        public bool IsActive = true;
        public ICollider? Collider;

        public bool IsOpen = false;
    }

    public class Ball : GameObject3D
    {
        public Vector3 Velocity;
        public Vector3 Color = new Vector3(1, 1, 1); // white
        public float Radius = 0.2f;
        public SphereCollider SphereCollider;

        public Ball(Vector3 position, Vector3 velocity)
        {
            Position = position;
            Velocity = velocity;
            Scale = new Vector3(Radius * 2); // Cube is 1x1x1, so scale by 2*Radius to match sphere visual size
            SphereCollider = new SphereCollider(Position, Radius);
            Collider = SphereCollider;
        }

        public void Update(float dt)
        {
            Position += Velocity * dt;
            SphereCollider.Center = Position;
        }
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

    public class Game : GameWindow
    {
        private int shaderProgram;
        private int vaoCube, vaoGround;
        private int texGround, texCube;

        private Camera camera = new();
        private List<GameObject3D> walls = new();
        private GameObject3D door;
        private GameObject3D target;
        private GameObject3D playerObject;
        private List<Ball> balls = new();
        private List<GameObject3D> boxes = new();

        private bool firstMouse = true;
        private Vector2 lastMousePos;
        private float fov = 60f;
        private Matrix4 projection;

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

            // Walls
            walls.Add(new GameObject3D()
            {
                Position = new Vector3(-3, 0.5f, -4),
                Scale = new Vector3(1, 1, 0.2f),
                // Initializing a default AABB, it will be updated in UpdateFrame.
                Collider = new AABB3D(Vector3.Zero, Vector3.Zero),
                IsActive = true
            });

            // Door (black)
            door = new GameObject3D()
            {
                Position = new Vector3(0, 0.5f, -2),
                Scale = new Vector3(1.5f, 1.5f, 0.2f),
                Collider = new AABB3D(Vector3.Zero, Vector3.Zero),
                IsActive = true,
                IsOpen = false,
                Rotation = Vector3.Zero
            };

            // Target (green)
            target = new GameObject3D()
            {
                Position = new Vector3(2.5f, 0.5f, -2),
                Scale = new Vector3(1.5f, 1.5f, 0.2f),
                Collider = new AABB3D(Vector3.Zero, Vector3.Zero),
                IsActive = true,
                IsOpen = false
            };

            // Player
            playerObject = new GameObject3D()
            {
                Position = new Vector3(0f, 0.9f, 3f), // Lifted slightly so it stands on the ground
                Scale = new Vector3(0.3f, 1.8f, 0.3f),
                // Initializing a default AABB, it will be updated in UpdateFrame.
                Collider = new AABB3D(Vector3.Zero, Vector3.Zero),
                IsActive = true
            };

            // Camera position is set to first-person view in OnUpdateFrame
            camera.Position = playerObject.Position + new Vector3(0, 1.7f, 0);


            // Boxes (32 breakable)
            int gridSize = 8;
            float spacing = 2.0f;
            for (int i = 0; i < 32; i++)
            {
                int row = i / gridSize;
                int col = i % gridSize;
                var box = new GameObject3D()
                {
                    Position = new Vector3(-7 + col * spacing, 0.5f, -6 - row * spacing),
                    Scale = new Vector3(1, 1, 1),
                    // Initializing a default AABB, it will be updated in UpdateFrame.
                    Collider = new AABB3D(Vector3.Zero, Vector3.Zero),
                    IsActive = true
                };
                boxes.Add(box);
            }

            vaoGround = CreateMesh(ShapeFactory.CreatePlaneTextured(100f)); // Made ground larger
            vaoCube = CreateMesh(ShapeFactory.CreateCubeTextured());
            texCube = TextureLoader.LoadTexture();
            texGround = TextureLoader.LoadTexture();

            var aspect = Size.X / (float)Size.Y;
            projection = Matrix4.CreatePerspectiveFieldOfView(MathHelper.DegreesToRadians(fov), aspect, 0.1f, 100f);
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

        /// <summary>
        /// Helper to update the AABB bounds for a list of GameObjects based on their Position and Scale.
        /// </summary>
        private void UpdateColliderBounds(List<GameObject3D> objects)
        {
            foreach (var obj in objects)
            {
                // Only update active objects with AABB colliders
                if (obj.Collider is AABB3D box && obj.IsActive)
                {
                    // AABB Min/Max are calculated as WorldPosition +/- HalfScale
                    Vector3 halfScale = obj.Scale * 0.5f;
                    box.Min = obj.Position - halfScale;
                    box.Max = obj.Position + halfScale;
                }
            }
        }

        protected override void OnUpdateFrame(FrameEventArgs e)
        {
            base.OnUpdateFrame(e);

            var kb = KeyboardState;
            var ms = MouseState;

            float dt = (float)e.Time;

            if (kb.IsKeyDown(Keys.Escape))
                Close();

            // --- Mouse Look ---
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

            // --- Player Movement ---
            Vector3 dir = Vector3.Zero;
            Vector3 forward = camera.Front;
            forward.Y = 0;
            forward = Vector3.Normalize(forward);
            Vector3 right = Vector3.Normalize(Vector3.Cross(forward, Vector3.UnitY));

            if (kb.IsKeyDown(Keys.W)) dir += forward;
            if (kb.IsKeyDown(Keys.S)) dir -= forward;
            if (kb.IsKeyDown(Keys.A)) dir -= right;
            if (kb.IsKeyDown(Keys.D)) dir += right;
            if (dir.LengthSquared > 0) dir = dir.Normalized();

            Vector3 newPlayerPosition = playerObject.Position + dir * 3f * dt;

            // --- Update all AABB colliders based on their world position ---
            UpdateColliderBounds(walls);
            UpdateColliderBounds(boxes);
            UpdateColliderBounds(new List<GameObject3D> { door, target });

            // Update player collider for movement
            if (playerObject.Collider is AABB3D playerBox)
            {
                // Test collision at the new position
                Vector3 halfScale = playerObject.Scale * 0.5f;
                playerBox.Min = newPlayerPosition - halfScale;
                playerBox.Max = newPlayerPosition + halfScale;

                bool collided = false;

                // Check collision with walls and boxes
                foreach (var wall in walls)
                {
                    if (wall.IsActive && wall.Collider!.Intersects(playerBox))
                    {
                        collided = true;
                        break;
                    }
                }

                if (!collided)
                {
                    foreach (var box in boxes)
                    {
                        if (box.IsActive && box.Collider!.Intersects(playerBox))
                        {
                            collided = true;
                            break;
                        }
                    }
                }
             
                if (!collided && target.IsActive)
                {
                    if (target.Collider!.Intersects(playerBox))
                    {
                        collided = true;
                    }
                }
                if (!collided && door.IsActive && !door.IsOpen) // Only check door if it's closed
                {
                    if (door.Collider!.Intersects(playerBox))
                    {
                        collided = true;
                    }
                }

                // If no collision, update player position and collider
                if (!collided)
                {
                    playerObject.Position = newPlayerPosition;
                    // Final update of the actual collider's bounds after movement
                    playerBox.Min = playerObject.Position - halfScale;
                    playerBox.Max = playerObject.Position + halfScale;
                }
            }


            // Camera is fixed to player's head (first-person view)
            camera.Position = playerObject.Position + new Vector3(0, 1.7f, 0);


            // Door toggle and smooth rotation
            if (kb.IsKeyPressed(Keys.E))
            {
                // Calculate distance using only XZ plane for interaction
                Vector2 doorXZ = new Vector2(door.Position.X, door.Position.Z);
                Vector2 playerXZ = new Vector2(playerObject.Position.X, playerObject.Position.Z);
                float distance = (doorXZ - playerXZ).Length;

                if (distance < 2.5f)
                {
                    door.IsOpen = !door.IsOpen;
                }
            }

            float rotationSpeed = 2f; // radians per second
            float targetAngle = door.IsOpen ? MathHelper.PiOver2 : 0f; // 90 degrees open
            float angleDiff = targetAngle - door.Rotation.Y;
            float rotationStep = rotationSpeed * dt;

            if (Math.Abs(angleDiff) > 0.01f)
            {
                door.Rotation = new Vector3(0, door.Rotation.Y + Math.Sign(angleDiff) * Math.Min(rotationStep, Math.Abs(angleDiff)), 0);
            }
            else
            {
                door.Rotation = new Vector3(0, targetAngle, 0);
            }

            // Throw balls with Spacebar, cooldown 0.5s
            if (throwCooldown > 0)
                throwCooldown -= dt;
            if (kb.IsKeyDown(Keys.Space) && throwCooldown <= 0)
            {
                Vector3 ballPos = camera.Position + camera.Front * (playerObject.Scale.X + 0.1f);
                Vector3 ballVelocity = camera.Front * 20f; // Increased speed for effect
                balls.Add(new Ball(ballPos, ballVelocity));
                throwCooldown = 0.5f;
            }

            // Update balls and collision with target and boxes
            for (int i = balls.Count - 1; i >= 0; i--)
            {
                Ball currentBall = balls[i]; // Use a local variable for clarity
                currentBall.Update(dt);

                // Simple gravity and ground bounce
                currentBall.Velocity.Y -= 9.8f * dt;
                if (currentBall.Position.Y < currentBall.Radius)
                {
                    currentBall.Position.Y = currentBall.Radius;
                    currentBall.Velocity.Y *= -0.7f; // Bounce with dampening

                    if (currentBall.Velocity.LengthSquared < 1.0f)
                    {
                        balls.RemoveAt(i);
                        continue;
                    }
                }

                // Removal check if ball falls way off the map
                if (currentBall.Position.Y < -10)
                {
                    balls.RemoveAt(i);
                    continue;
                }

                // 1. Check collision with the Green Target Wall (Bounce Logic)
                if (target.IsActive && target.Collider is AABB3D targetBox && currentBall.SphereCollider.Intersects(targetBox))
                {
                    // Determine collision based on wall's Z-axis (front/back faces)
                    Vector3 centerToCenter = currentBall.Position - target.Position;

                    // Assuming the target is a thin wall primarily along the Z-axis
                    if (Math.Abs(centerToCenter.Z) > Math.Abs(centerToCenter.X) && targetBox.Max.Z - targetBox.Min.Z < 0.5f)
                    {
                        // Reflect the Z velocity and dampen the bounce
                        currentBall.Velocity.Z *= -1f;
                        currentBall.Velocity *= 0.8f;

                        // Nudge the ball out of the wall
                        float penetrationDepth = (currentBall.Radius + target.Scale.Z / 2.0f) - Math.Abs(centerToCenter.Z);
                        currentBall.Position.Z += Math.Sign(centerToCenter.Z) * penetrationDepth;
                    }
                    // Skip other checks as the ball bounced
                    continue;
                }

                bool ballRemoved = false;
                // 2. Check collision with breakable boxes
                for (int j = boxes.Count - 1; j >= 0; j--)
                {
                    // Check only active boxes
                    if (boxes[j].IsActive && boxes[j].Collider != null && currentBall.SphereCollider.Intersects(boxes[j].Collider))
                    {
                        // BREAK THE BOX
                        boxes[j].IsActive = false;
                        balls.RemoveAt(i);
                        ballRemoved = true;
                        break;
                    }
                }
                if (ballRemoved) continue;
            }
        }

        protected override void OnRenderFrame(FrameEventArgs e)
        {
            base.OnRenderFrame(e);

            GL.Clear(ClearBufferMask.ColorBufferBit | ClearBufferMask.DepthBufferBit);
            GL.UseProgram(shaderProgram);

            // Set uniforms
            GL.Uniform3(GL.GetUniformLocation(shaderProgram, "viewPos"), ref camera.Position);
            GL.Uniform3(GL.GetUniformLocation(shaderProgram, "lightPos"), new Vector3(0, 8, 4));
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
            GL.Uniform3(GL.GetUniformLocation(shaderProgram, "matDiffuse"), new Vector3(0.4f, 0.3f, 0.2f));
            var groundModel = Matrix4.CreateTranslation(0, -0.01f, 0);
            GL.UniformMatrix4(GL.GetUniformLocation(shaderProgram, "model"), false, ref groundModel);
            GL.DrawArrays(PrimitiveType.Triangles, 0, 6);

            // Draw walls gray
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

            // Draw door black with rotation
            if (door.IsActive)
            {
                GL.Uniform3(GL.GetUniformLocation(shaderProgram, "matDiffuse"), new Vector3(0.1f, 0.1f, 0.1f));

                // Rotation transformation adjusted to use the right side as a hinge (assuming X-axis alignment).
                float hingeOffset = door.Scale.X * 0.5f;
                var doorModel = Matrix4.CreateScale(door.Scale) *
                                Matrix4.CreateTranslation(hingeOffset, 0, 0) * Matrix4.CreateRotationY(door.Rotation.Y) * Matrix4.CreateTranslation(-hingeOffset, 0, 0) * Matrix4.CreateTranslation(door.Position);

                GL.UniformMatrix4(GL.GetUniformLocation(shaderProgram, "model"), false, ref doorModel);
                GL.DrawArrays(PrimitiveType.Triangles, 0, 36);
            }

            // Draw target green
            if (target.IsActive)
            {
                GL.Uniform3(GL.GetUniformLocation(shaderProgram, "matDiffuse"), new Vector3(0f, 0.8f, 0.1f));
                var targetModel = Matrix4.CreateScale(target.Scale) * Matrix4.CreateTranslation(target.Position);
                GL.UniformMatrix4(GL.GetUniformLocation(shaderProgram, "model"), false, ref targetModel);
                GL.DrawArrays(PrimitiveType.Triangles, 0, 36);
            }

            // Draw boxes brown
            GL.Uniform3(GL.GetUniformLocation(shaderProgram, "matDiffuse"), new Vector3(0.6f, 0.3f, 0.1f));
            foreach (var box in boxes)
            {
                if (!box.IsActive) continue;
                var boxModel = Matrix4.CreateScale(box.Scale) * Matrix4.CreateTranslation(box.Position);
                GL.UniformMatrix4(GL.GetUniformLocation(shaderProgram, "model"), false, ref boxModel);
                GL.DrawArrays(PrimitiveType.Triangles, 0, 36);
            }

            // Draw player blue
            GL.Uniform3(GL.GetUniformLocation(shaderProgram, "matDiffuse"), new Vector3(0.2f, 0.4f, 0.7f));
            var playerModel = Matrix4.CreateScale(playerObject.Scale) * Matrix4.CreateTranslation(playerObject.Position);
            GL.UniformMatrix4(GL.GetUniformLocation(shaderProgram, "model"), false, ref playerModel);
            GL.DrawArrays(PrimitiveType.Triangles, 0, 36);

            // Draw balls white
            GL.Uniform3(GL.GetUniformLocation(shaderProgram, "matDiffuse"), new Vector3(1.0f));
            foreach (var ball in balls)
            {
                GL.Uniform3(GL.GetUniformLocation(shaderProgram, "matDiffuse"), ball.Color);
                var ballModel = Matrix4.CreateScale(ball.Scale) * Matrix4.CreateTranslation(ball.Position);
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
}";

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
}";
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
                -s,-s,-s, -1,0,0, 0,1,
                -s,-s,-s, -1,0,0, 0,1,
                -s,-s, s, -1,0,0, 0,0,
                -s, s, s, -1,0,0, 1,0,
                // right face
                 s, s, s, 1,0,0, 1,0,
                 s, s,-s, 1,0,0, 1,1,
                 s,-s,-s, 1,0,0, 0,1,
                 s,-s,-s, 1,0,0, 0,1,
                 s,-s, s, 1,0,0, 0,0,
                 s, s, s, 1,0,0, 1,0,
                // bottom face
                -s,-s,-s, 0,-1,0, 0,1,
                 s,-s,-s, 0,-1,0, 1,1,
                 s,-s, s, 0,-1,0, 1,0,
                 s,-s, s, 0,-1,0, 1,0,
                -s,-s, s, 0,-1,0, 0,0,
                -s,-s,-s, 0,-1,0, 0,1,
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
                -s, 0, -s, 0,1,0, 0, size,
                 s, 0, -s, 0,1,0, size, 0,
                 s, 0,  s, 0,1,0, size, size,
                 s, 0,  s, 0,1,0, size, size,
                -s, 0,  s, 0,1,0, 0, size,
                -s, 0, -s, 0,1,0, 0, 0,
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
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.LinearMipmapLinear);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Linear);
            GL.GenerateMipmap(GenerateMipmapTarget.Texture2D);
            return tex;
        }
    }

    public static class Program
    {
        public static void Main()
        {
            var nativeWindowSettings = new NativeWindowSettings()
            {
                Size = new Vector2i(1280, 720),
                Title = "OpenTK Door Rotation and Breakable Boxes"
            };

            using var window = new Game(GameWindowSettings.Default, nativeWindowSettings);
            window.Run();
        }
    }
}