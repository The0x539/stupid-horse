#version 450

precision mediump float;

layout(binding = 0) uniform stuff {
	float time;
	vec2 click_pos;
	vec2 window_dims;
};

layout(location = 0) in vec2 a_position;
layout(location = 1) in vec4 a_color;

layout(location = 0) out vec4 v_color;

void main() {
	float r = length(a_position);
	float theta = atan(a_position.y, a_position.x);
	theta += time/10;
	vec2 pos = vec2(r*cos(theta), r*sin(theta));

	float w = window_dims.x;
	float h = window_dims.y;
	pos.x *= min(h/w, 1.0);
	pos.y *= min(w/h, 1.0);

	pos += click_pos;

	gl_Position = vec4(pos.xy, 0.0, 1.0);
	v_color = a_color;
}
