#version 450

precision mediump float;

layout(binding = 0) uniform stuff {
	vec2 click_pos;
	float time;
	float scale;
};

layout(location = 0) in vec2 a_position;
layout(location = 1) in vec2 a_coords;

layout(location = 0) out vec2 v_coords;

void main() {
	float r = length(a_position) * scale;
	float theta = atan(a_position.y, a_position.x);
	theta += time/10;
	vec2 pos = vec2(r*cos(theta), r*sin(theta));

	pos += click_pos;

	gl_Position = vec4(pos.xy, 0.0, 1.0);
	v_coords = a_coords;
}
