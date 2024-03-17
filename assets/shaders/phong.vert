#version 330 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;

uniform mat4 _modelMatrix;
uniform mat4 _viewMatrix;
uniform mat4 _projMatrix;

out vec3 fragNormal;
out vec3 fragPosition;


void main()
{ 
	//fragPosition = vec3(model * vec4(aPos, 1.0));
	//fragNormal = mat3(transpose(inverse(model))) * aNormal;  
	fragPosition = aPos;
	fragNormal = aNormal;

	gl_Position = _projMatrix* _viewMatrix *  _modelMatrix * vec4(aPos, 1.0);
}