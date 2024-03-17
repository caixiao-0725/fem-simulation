#version 330 core	// 指定GLSL版本
// 定义变量
in vec3 aPos;
//in vec3 aColor;
out vec3 FragPos;

uniform mat4 _modelMatrix;
uniform mat4 _viewMatrix;
uniform mat4 _projMatrix;

// 主函数
void main()
{
    gl_Position = _projMatrix* _viewMatrix *  _modelMatrix *vec4(aPos, 1.0);	// 设置顶点坐标
    FragPos = aPos;
}