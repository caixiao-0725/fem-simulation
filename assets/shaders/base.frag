#version 330 core
// 定义变量
in vec3 FragPos;
out vec4 FragColor;

uniform vec3 viewPos;
uniform vec3 light_pos;

// 主函数
void main()
{
    //light_pos = vec3(1.0,1.0,1.0)

    FragColor = vec4(0.5, 1.0, 1.0, 1.0f);	// 设置像素颜色
} 
