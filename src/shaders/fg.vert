#version 450

precision mediump float;

layout(binding = 0) uniform stuff {
	// beware of alignment discrepancies between GLSL and Rust
	vec2 click_pos;
	vec2 dims;
	float time;
};

layout(location = 0) in vec2 a_position;
layout(location = 1) in vec3 a_color;

layout(location = 0) out vec3 v_color;

void main() {
	float r = length(a_position);
	float theta = atan(a_position.y, a_position.x);
	theta += time/10;
	vec2 pos = vec2(r*cos(theta), r*sin(theta));

	float aspect_ratio = dims.x / dims.y;
	pos.x /= max(aspect_ratio, 1.0); // if wide, shrink x
	pos.y *= min(aspect_ratio, 1.0); // if tall, shrink y

	pos += click_pos;

	gl_Position = vec4(pos.xy, 0.0, 1.0);
	v_color = a_color;
}
