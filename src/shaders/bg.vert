#version 450

precision mediump float;

layout(binding = 0) uniform stuff {
	vec2 dims;
};

layout(location = 0) in vec2 a_position;
layout(location = 1) in vec3 a_color;

layout(location = 0) out vec3 v_color;

void main() {
	vec2 pos = a_position;

	float aspect_ratio = dims.x / dims.y;
	pos.x /= max(aspect_ratio, 1.0); // if wide, shrink x
	pos.y *= min(aspect_ratio, 1.0); // if tall, shrink y

	gl_Position = vec4(pos.xy, 0.0, 1.0);
	v_color = a_color;
}
