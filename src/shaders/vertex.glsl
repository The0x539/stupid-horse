#version 450

precision mediump float;

layout(binding = 0) uniform stuff {
	float time;
};

layout(binding = 1) uniform other_stuff {
	vec2 click_pos;
};

layout(binding = 2) uniform more_stuff {
	vec2 window_dims;
};

layout(location = 0) in vec2 position;

void main() {
	float r = length(position);
	float theta = atan(position.y, position.x);
	//theta += time/10;
	vec2 pos = vec2(r*cos(theta), r*sin(theta));

	float w = window_dims.x;
	float h = window_dims.y;
	pos.x *= min(h/w, 1.0);
	pos.y *= min(w/h, 1.0);

	//pos += click_pos;

	gl_Position = vec4(pos.xy, 0.0, 1.0);
}
