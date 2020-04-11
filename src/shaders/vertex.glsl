#version 450

precision mediump float;

layout(binding = 0) uniform stuff {
	float time;
};

layout(binding = 1) uniform other_stuff {
	vec2 click_pos;
};

layout(location = 0) in vec2 position;

void main() {
	float r = length(position);
	float theta = atan(position.y, position.x);
	theta += time/10;
	vec2 pos = vec2(r*cos(theta), r*sin(theta));

	pos += click_pos;

	gl_Position = vec4(pos.xy, 0.0, 1.0);
}
