#version 450

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0, rgba8) uniform writeonly image2D img;

float lt(float x, float y) {
	return max(sign(y - x), 0.0);
}

void main() {
	vec2 norm_coords = (gl_GlobalInvocationID.xy + 0.5) / imageSize(img);
	vec2 c = (norm_coords - 0.5) * 2.0 - vec2(1.0, 0.0);

	vec2 z = vec2(0.0, 0.0);
	float j = 0.0;
#define DI 0.01
	for (float i = 0.0; i < 1.0; i += DI) {
		z = vec2(
			z.x*z.x - z.y*z.y + c.x,
			z.y*z.x + z.x*z.y + c.y
		);
		j += lt(length(z), 4.0) * DI;
	}

	vec4 to_write = vec4(vec3(j), 1.0);
	imageStore(img, ivec2(gl_GlobalInvocationID.xy), to_write);
}
