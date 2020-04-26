#version 450

layout(binding = 1) uniform sampler2D tex;

layout(location = 0) in vec2 v_coords;

layout(location = 0) out vec4 f_color;

void main() {
	f_color = texture(tex, v_coords);
}
