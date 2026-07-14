canvas = document.querySelector('canvas');
canvas.width = 600;
canvas.height = 400;

class Mesh {
    constructor(cellsX, cellsY, lengthX, lengthY) {
        this.cellsX = cellsX; this.cellsY = cellsY;
        this.nodesNumX = cellsX + 1; this.nodesNumY = cellsY + 1;
        this.lengthX = lengthX; this.lengthY = lengthY;
        this.dx = lengthX / cellsX; this.dy = lengthY / cellsY;
        this.nodesNum = this.nodesNumX * this.nodesNumY;

        this.nodes = Array.from({length: this.nodesNum}, (_, i) => ({
            idx: i, ix: Math.floor(i / this.nodesNumY), iy: i % this.nodesNumY,
            x: 0., y: 0., z: 0., w: 1., 
            neighbors: [], edges: []
        }));
        
        this.edges = Array.from({length: 3 * cellsX * cellsY + cellsX + cellsY}, (_, i) => ({
            idx: i, node1: null, node2: null, neighbors: [], color: -1,
        }));

        let eIdx = 0;
        for(let node of this.nodes) {
            node.x = -0.5 * lengthX + node.ix * this.dx;
            node.y = -0.5 * lengthY + node.iy * this.dy;
            if(node.ix + 1 < this.nodesNumX) {
                let nIdx = node.idx + this.nodesNumY;
                node.neighbors.push(this.nodes[nIdx]);
                this.nodes[nIdx].neighbors.push(node);
                
                this.edges[eIdx].node1 = node;
                node.edges.push(this.edges[eIdx]);
                this.edges[eIdx].node2 = this.nodes[nIdx];
                this.nodes[nIdx].edges.push(this.edges[eIdx]);
                ++eIdx;

                if(node.iy + 1 < this.nodesNumY) {
                    ++nIdx;
                    node.neighbors.push(this.nodes[nIdx]);
                    this.nodes[nIdx].neighbors.push(node);
                    
                    this.edges[eIdx].node1 = node;
                    node.edges.push(this.edges[eIdx]);
                    this.edges[eIdx].node2 = this.nodes[nIdx];
                    this.nodes[nIdx].edges.push(this.edges[eIdx]);
                    ++eIdx;
                }
            }
            if(node.iy + 1 < this.nodesNumY) {
                let nIdx = node.idx + 1;
                node.neighbors.push(this.nodes[nIdx]);
                this.nodes[nIdx].neighbors.push(node);
                
                this.edges[eIdx].node1 = node;
                node.edges.push(this.edges[eIdx]);
                this.edges[eIdx].node2 = this.nodes[nIdx];
                this.nodes[nIdx].edges.push(this.edges[eIdx]);
                ++eIdx;
            }
        }

        this.initPositions = new Float32Array(4 * this.nodesNum);
        for(let node of this.nodes) {
            let idx = 4 * node.idx;
            this.initPositions[idx++] = node.x;
            this.initPositions[idx++] = node.y;
            this.initPositions[idx++] = node.z;
            this.initPositions[idx++] = node.w;
        }

        for(let edge of this.edges) {
            for(let e of edge.node1.edges) {
                if(e != edge) {
                    edge.neighbors.push(e);
                }
            }
            for(let e of edge.node2.edges) {
                if(e != edge) {
                    edge.neighbors.push(e);
                }
            }
        }

        this.numColors = 0;
        for(let edge of this.edges) {
            let colors = new Array(20).fill(true);
            for(let e of edge.neighbors) {
                if(e.color >= 0) {
                    colors[e.color] = false;
                }
            }
            for(let i = 0; i < colors.length; ++i) {
                if(colors[i]) {
                    edge.color = i;
                    break;
                }
            }
            this.numColors = Math.max(this.numColors, edge.color);
        }
        ++this.numColors;

        let colorGroups = Array.from({length: this.numColors}, (_, color) => (this.edges.filter(e => e.color == color)));
        this.colorGroupOffsets = new Uint32Array(Math.max(this.numColors, 10));
        this.colorGroupSizes = new Uint32Array(Math.max(this.numColors, 10));
        this.wireframeIndices = new Uint32Array(2 * this.edges.length)
        let pIdx = 0;
        for(let i = 0; i < this.numColors; ++i) {
            this.colorGroupSizes[i] = 2 * colorGroups[i].length;
            if(i > 0) {
                this.colorGroupOffsets[i] = this.colorGroupOffsets[i - 1] + this.colorGroupSizes[i-1];
            }
            for(let edge of colorGroups[i]) {
                this.wireframeIndices[pIdx++] = edge.node1.idx;
                this.wireframeIndices[pIdx++] = edge.node2.idx;
            }
        }

        this.triangleIndices = new Uint32Array(6 * cellsX * cellsY);    
        for(let ix = 0; ix < cellsX; ++ix) {
            for(let iy = 0; iy < cellsY; ++iy) {
                let idx = 6 * (ix * cellsY + iy);
                let pIdx = ix * this.nodesNumY + iy;
                this.triangleIndices[idx++] = pIdx;
                this.triangleIndices[idx++] = pIdx + 1;
                this.triangleIndices[idx++] = pIdx + this.nodesNumY + 1;
                
                this.triangleIndices[idx++] = pIdx;
                this.triangleIndices[idx++] = pIdx + this.nodesNumY;
                this.triangleIndices[idx++] = pIdx + this.nodesNumY + 1;
            }
        }
    }
}

class Model {
    createGPUBuffer(cpuArray, usageFlags) {
        let gpuBuffer = this.device.createBuffer({size: cpuArray.byteLength, usage: usageFlags });
        this.device.queue.writeBuffer(gpuBuffer, 0, cpuArray);
        return gpuBuffer;
    }

    async initGPU() {
        try {
            let adapter = await navigator.gpu.requestAdapter();
            if (!adapter) {
                document.getElementById('status').textContent = 'WebGPU не поддерживается';
                return;
            }
            this.device = await adapter.requestDevice();

            this.context = canvas.getContext('webgpu');
            this.format = navigator.gpu.getPreferredCanvasFormat();
            this.context.configure({ device: this.device, format: this.format });
        } catch (error) {
            console.error('Error:', error);
            document.getElementById('status').textContent = 'Error: ' + error.message;
            document.getElementById('status').style.color = 'red';
        }
    }

    async initRenderShader() {
        let renderShaderCode = `
            struct CameraParams {
                camAngle: f32,
                camDistance: f32,
                camHeight: f32,
            };

            struct VertexOutput {
                @builtin(position) position: vec4<f32>,
                @location(0) color: vec4<f32>,
            }
            
            @group(0) @binding(0) var<uniform> params: CameraParams;
            @group(0) @binding(1) var<uniform> colorGroupSizes: array<u32, 10>;
            @group(0) @binding(2) var<uniform> colorGroupOffsets: array<u32, 10>;
            @group(0) @binding(3) var<storage, read> vertices: array<vec4f>;
            @group(0) @binding(4) var<storage, read> triangleIndices: array<u32>;
            @group(0) @binding(5) var<storage, read> wireframeIndices: array<u32>;

            fn rotate(pos : vec4f) -> vec4f {
                let angle = params.camAngle;
                let distance = params.camDistance;
                let height = params.camHeight;

                let camX = distance * cos(angle);
                let camY = distance * sin(angle);
                let camZ = height;
                
                let vCam = vec3f(camX, camY, camZ);
                let tar = vec3f(0.0, 0.0, 0.0);
                let forward = normalize(tar - vCam);
                let right = normalize(cross(forward, vec3f(0.0, 0.0, 1.0)));
                let up = normalize(cross(right, forward));
                
                let viewMatrix = mat4x4f(
                    vec4f(right.x, up.x, -forward.x, 0.0),
                    vec4f(right.y, up.y, -forward.y, 0.0),
                    vec4f(right.z, up.z, -forward.z, 0.0),
                    vec4f(-dot(right, vCam), -dot(up, vCam), dot(forward, vCam), 1.0)
                );
                
                let aspect = 800.0 / 600.0;
                let fov = 1.0;
                let near = 0.1;
                let far = 100.0;
                let f = 1.0 / tan(fov / 2.0);
                let projMatrix = mat4x4f(
                    vec4f(f / aspect, 0.0, 0.0, 0.0),
                    vec4f(0.0, f, 0.0, 0.0),
                    vec4f(0.0, 0.0, (far + near) / (near - far), -1.0),
                    vec4f(0.0, 0.0, (2.0 * far * near) / (near - far), 0.0)
                );
                
                let viewPos = viewMatrix * pos;
                let clipPos = projMatrix * viewPos;
                    
                return clipPos;
            }

            @vertex
            fn vs_main(
            @builtin(vertex_index) vertexIndex : u32) -> @builtin(position) vec4f {     
                return rotate(vertices[triangleIndices[vertexIndex]]);
            }

            @vertex
            fn vs_wire(
                @builtin(vertex_index) vertexIndex : u32, 
                @builtin(instance_index) offset_id : u32) -> @builtin(position) vec4f {
                return rotate(vertices[wireframeIndices[vertexIndex]]);
            }           
            
            @fragment
            fn fs_main() -> @location(0) vec4f {
                return vec4f(0.5, 0.5, 0.5, 1.0);
            }

            @fragment
            fn fs_wireframe() -> @location(0) vec4f {
                return vec4f(0., 0., 0., 1.);
            }

            @vertex
            fn vs_wire_color(
                @builtin(vertex_index) vertexIndex : u32, 
                @builtin(instance_index) offset_id : u32) -> VertexOutput 
            {
                let colors = array( vec4f(1., 0., 0., 1.), vec4f(0., 1., 0., 1.), vec4f(0., 0., 1., 1.), 
                    vec4f(1., 1., 0., 1.), vec4f(1., 0., 1., 1.), vec4f(0., 1., 1., 1.), 
                    vec4f(0., 0., 0., 1.), vec4f(1., 1., 1., 1.));

                var output: VertexOutput;
                output.position = rotate(vertices[wireframeIndices[vertexIndex]]);
                for(var i = 0; i < 10; i = i + 1) {
                    if(vertexIndex >= colorGroupOffsets[i] && vertexIndex < colorGroupOffsets[i] + colorGroupSizes[i]) {
                        output.color = colors[i];
                        break;
                    }
                }
                return output;
            }

            @fragment
            fn fs_wireframe_color(input: VertexOutput) -> @location(0) vec4f {
                return input.color;
            }
        `;
        let renderModule = this.device.createShaderModule({ code: renderShaderCode });

        let info = await renderModule.getCompilationInfo();
        if (info.messages.length > 0) {
            console.warn('Shader warn:', info.messages);
        }

        // Pipelines
        let bindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform", }, },
                { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform", }, },
                { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform", }, },
                { binding: 3, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage", }, },
                { binding: 4, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage", }, },
                { binding: 5, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage", }, },
            ],
        });
        let pipelineLayout = this.device.createPipelineLayout({
            bindGroupLayouts: [bindGroupLayout],
        });
        this.facesPipeline = this.device.createRenderPipeline({
            layout: pipelineLayout,
            vertex: {
                module: renderModule,
                entryPoint: 'vs_main',
                buffers: [],
            },
            fragment: {
                module: renderModule,
                entryPoint: 'fs_main',
                targets: [{ format: this.format }],
            },
            primitive: {
                topology: 'triangle-list',
            },
        });
        this.facesBindGroup = this.device.createBindGroup({
            layout: bindGroupLayout,
            entries: [             
                { binding: 0, resource: this.camBuffer},
                { binding: 1, resource: this.colorGroupSizeBuffer},
                { binding: 2, resource: this.colorGroupOffsetBuffer},
                { binding: 3, resource: this.positionBuffer},
                { binding: 4, resource: this.triangleBuffer},
                { binding: 5, resource: this.wireframeBuffer},
            ],
        });
        this.wireframePipeline = this.device.createRenderPipeline({
            layout: pipelineLayout,
            vertex: {
                module: renderModule,
                entryPoint: 'vs_wire',
                buffers: [],
            },
            fragment: {
                module: renderModule,
                entryPoint: 'fs_wireframe',
                targets: [{ format: this.format }],
            },
            primitive: {
                topology: 'line-list',
            },
        });

        
        this.wireframeColorPipeline = this.device.createRenderPipeline({
            layout: pipelineLayout,
            vertex: {
                module: renderModule,
                entryPoint: 'vs_wire_color',
                buffers: [],
            },
            fragment: {
                module: renderModule,
                entryPoint: 'fs_wireframe_color',
                targets: [{ format: this.format }],
            },
            primitive: {
                topology: 'line-list',
            },
        });

        this.wireframeBindGroup = this.device.createBindGroup({
            layout: bindGroupLayout,        
            entries: [
                { binding: 0, resource: this.camBuffer},
                { binding: 1, resource: this.colorGroupSizeBuffer},
                { binding: 2, resource: this.colorGroupOffsetBuffer},
                { binding: 3, resource: this.positionBuffer},
                { binding: 4, resource: this.triangleBuffer},
                { binding: 5, resource: this.wireframeBuffer},
            ],
        });    
    }

    async initComputeShader() {
        let computeShaderCode = `
            const TWO_PI = 6.283185307179586;
            struct SimParams {
                nodesX: u32,
                nodesY: u32,
                dt: f32,
                dx: f32,
                dy: f32,
                elasticK: f32,
                gravityG: f32,
                omega: f32,
                amplitude: f32,
            };

            @group(0) @binding(0) var<uniform> params: SimParams;
            @group(0) @binding(1) var<uniform> colorGroupSizes: array<u32, 10>;
            @group(0) @binding(2) var<uniform> colorGroupOffsets: array<u32, 10>;
            @group(0) @binding(3) var<storage, read_write> pos: array<vec4f>;
            @group(0) @binding(4) var<storage, read_write> newPos: array<vec4f>;
            @group(0) @binding(5) var<storage, read_write> velocities: array<vec4f>;
            @group(0) @binding(6) var<storage, read_write> invMasses: array<f32>;
            @group(0) @binding(7) var<storage, read_write> phase: f32;
            @group(0) @binding(8) var<storage, read> wireframeIndices: array<u32>;
            @group(0) @binding(9) var<storage, read> colorIdx: u32;

            @compute @workgroup_size(64)
            fn predict(@builtin(global_invocation_id) id: vec3<u32>) {
                let vIdx = id.x;
                let ix = vIdx / params.nodesY;
                let iy = vIdx % params.nodesY;

                newPos[vIdx] = pos[vIdx];

                if((ix == 0 && (iy == 0 || (iy + 1) == params.nodesY)) || (
                    (ix + 1) == params.nodesX && (iy == 0 || (iy + 1) == params.nodesY)) ) {
                        return; // corners
                }
                else if(ix == (params.nodesX/2) && iy == (params.nodesY/2) && params.omega >= 0.0) {
                    phase += params.omega * params.dt;
                    if(phase > TWO_PI) {
                        phase -= TWO_PI;
                    }
                    newPos[vIdx][2] = params.amplitude * sin(phase);   
                }
                else {
                    velocities[vIdx][2] -= params.gravityG * params.dt;                
                    newPos[vIdx] += velocities[vIdx] * params.dt;
                    newPos[vIdx][3] = 1.;
                }           
            }
            
            @compute @workgroup_size(64)
            fn project(@builtin(global_invocation_id) id: vec3<u32>) {
                let vIdx = id.x;
                if(2 * vIdx < colorGroupSizes[colorIdx]) {
                    let vIdx1 = wireframeIndices[colorGroupOffsets[colorIdx] + 2 * vIdx];
                    let vIdx2 = wireframeIndices[colorGroupOffsets[colorIdx] + 2 * vIdx + 1];
                    var dl0 = 0.;
                    if(vIdx1 + 1 == vIdx2 || vIdx2 + 1 == vIdx1) {
                        dl0 = params.dy;
                    }
                    else if(vIdx1 + params.nodesY == vIdx2 || vIdx2 + params.nodesY == vIdx1) {
                        dl0 = params.dx;
                    }
                    else {
                        dl0 = sqrt(params.dx * params.dx + params.dy * params.dy);
                    }
                    var dr = newPos[vIdx1] - newPos[vIdx2];
                    dr = params.elasticK / (invMasses[vIdx1] + invMasses[vIdx2]) * 
                        (length(dr) - dl0) * normalize(dr);
                    newPos[vIdx1] -= invMasses[vIdx1] * dr;
                    newPos[vIdx2] += invMasses[vIdx2] * dr;
                }
            }

            @compute @workgroup_size(64)
            fn update(@builtin(global_invocation_id) id: vec3<u32>) {
                let vIdx = id.x;
                velocities[vIdx] = (newPos[vIdx] - pos[vIdx]) / params.dt;
                pos[vIdx] = newPos[vIdx];
            }
        `;
        let computeModule = this.device.createShaderModule({ code: computeShaderCode });
        let info = await computeModule.getCompilationInfo();
        if (info.messages.length > 0) {
            console.warn('Shader warn:', info.messages);
        }
        // Pipelines
        let bindGroupLayout = this.device.createBindGroupLayout({
            entries: [ { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform", }, },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform", }, },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform", }, },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage", }, },
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage", }, },
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage", }, },
                { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage", }, },
                { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage", }, },
                { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage", }, },
                { binding: 9, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage", }, },
            ],
        });
        let pipelineLayout = this.device.createPipelineLayout({
            bindGroupLayouts: [bindGroupLayout],
        });

        this.computeBindGroup = this.device.createBindGroup({
            layout: bindGroupLayout,
            entries: [
                { binding: 0, resource: this.simBuffer},
                { binding: 1, resource: this.colorGroupSizeBuffer},
                { binding: 2, resource: this.colorGroupOffsetBuffer},
                { binding: 3, resource: this.positionBuffer},
                { binding: 4, resource: this.newPosBuffer},
                { binding: 5, resource: this.velocityBuffer},
                { binding: 6, resource: this.invMassBuffer},
                { binding: 7, resource: this.phaseBuffer},
                { binding: 8, resource: this.wireframeBuffer},
                { binding: 9, resource: this.colorIdxBuffer},
            ],
        });
        
        this.predictionPipeline = this.device.createComputePipeline({
            layout: pipelineLayout,
            compute: {
                module: computeModule,
                entryPoint: 'predict',
            },
        });

        this.projectionPipeline = this.device.createComputePipeline({
            layout: pipelineLayout,
            compute: {
                module: computeModule,
                entryPoint: 'project',
            },
        });

        this.updatePipeline = this.device.createComputePipeline({
            layout: pipelineLayout,
            compute: {
                module: computeModule,
                entryPoint: 'update',
            },
        });
    }

    render() {
        let commandEncoder = this.device.createCommandEncoder();
        let textureView = this.context.getCurrentTexture().createView();
        let pass = commandEncoder.beginRenderPass({
            colorAttachments: [{
                view: textureView,
                clearValue: { r: 0.1, g: 0.1, b: 0.3, a: 1.0 },
                loadOp: 'clear',
                storeOp: 'store',
            }]
        });
        // faces
        pass.setPipeline(this.facesPipeline);
        pass.setBindGroup(0, this.facesBindGroup);
        pass.draw(this.mesh.triangleIndices.length);

        // wireframe
        if(this.colorFlag) {
            pass.setPipeline(this.wireframeColorPipeline);
        }
        else {
            pass.setPipeline(this.wireframePipeline);
        }
        pass.setBindGroup(0, this.wireframeBindGroup);
        pass.draw(this.mesh.wireframeIndices.length);
        pass.end();
        this.device.queue.submit([commandEncoder.finish()]);
    }

    async compute() {
        let commandEncoder = this.device.createCommandEncoder();    
        // Prediction
        let computePass = commandEncoder.beginComputePass();
        computePass.setPipeline(this.predictionPipeline);
        computePass.setBindGroup(0, this.computeBindGroup);
        computePass.dispatchWorkgroups(Math.ceil(this.positions.length / 4 / 64));
        computePass.end();        
        this.device.queue.submit([commandEncoder.finish()]);

        // Projection
        for(let i = 0; i < 4; ++i) {
            for(let color = 0; color < this.mesh.numColors; ++color) {
                commandEncoder = this.device.createCommandEncoder();
                computePass = commandEncoder.beginComputePass();
                computePass.setBindGroup(0, this.computeBindGroup);
                computePass.setPipeline(this.projectionPipeline);
                this.setColorIndex(color);
                //await this.device.queue.onSubmittedWorkDone();
                computePass.dispatchWorkgroups(Math.ceil(this.mesh.colorGroupSizes[color] / 2 / 64));;
                computePass.end();
                this.device.queue.submit([commandEncoder.finish()]);
                //await this.device.queue.onSubmittedWorkDone()
            }
        }
        
        // Update values
        commandEncoder = this.device.createCommandEncoder();
        computePass = commandEncoder.beginComputePass();
        computePass.setBindGroup(0, this.computeBindGroup);
        computePass.setPipeline(this.updatePipeline);
        computePass.dispatchWorkgroups(Math.ceil(this.positions.length / 4 / 64));
        computePass.end();
        this.device.queue.submit([commandEncoder.finish()]);
    }

    updateCameraParams() {        
        let cameraParams = new Float32Array([this.camera.angle, this.camera.distance, this.camera.height]);
        this.device.queue.writeBuffer(this.camBuffer, 0, cameraParams);
        this.render();
    }

    setColorIndex(color) {
        let colorIdx = new Uint32Array([color]);
        this.device.queue.writeBuffer(this.colorIdxBuffer, 0, colorIdx);
    }

    async init(params) {
        document.getElementById('runBtn').disabled = true;
        this.simParam = {dt: params['dt'], elasticK: params['elasticK'], gravityG: params['gravityG'],
            omega: params['omega'], amplitude: params['amplitude']
        };

        this.mesh = new Mesh(params['cellsX'], params['cellsY'], params['lengthX'], params['lengthY']);
        this.positions = this.mesh.initPositions;
        this.velocities = new Float32Array(4 * this.mesh.nodesNum);
        this.invMasses = new Float32Array(this.mesh.nodesNum);
        const midIdx = Math.floor(this.mesh.nodesNumX / 2) * this.mesh.nodesNumY + Math.floor(this.mesh.nodesNumY / 2);    
        for(let node of this.mesh.nodes) {
            let idx = node.idx;
            if(node.ix == 0 || node.ix == this.mesh.cellsX) {
                if(node.iy == 0 || node.iy == this.mesh.cellsY) {
                    this.invMasses[idx] = 0.; // corner node is infinitely heavy
                }
                else {
                    this.invMasses[idx] = 2.; // side node is lighter
                }
            }
            else if(node.iy == 0 || node.iy == this.mesh.cellsY) {
                    this.invMasses[idx] = 2.; // side node is lighter
            }
            else {
                this.invMasses[idx] = 1.; // inner node
            }
            if(idx == midIdx && params['omega'] >= 0) {
                this.invMasses[idx] = 0.; // if oscillation is on, middle node is moved externally infinitely heavy
            }
        }

        await this.initGPU();

        let usageFlags = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
        this.positionBuffer = this.createGPUBuffer(this.positions, usageFlags);
        this.newPosBuffer = this.createGPUBuffer(this.positions, usageFlags);
        this.velocityBuffer = this.createGPUBuffer(this.velocities, usageFlags);
        this.invMassBuffer = this.createGPUBuffer(this.invMasses, usageFlags);
        let colorIdx = new Uint32Array([0]);
        this.colorIdxBuffer = this.createGPUBuffer(colorIdx, usageFlags);
        let phase = new Float32Array([0.]);
        this.phaseBuffer = this.createGPUBuffer(phase, usageFlags);
        
        usageFlags =  GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
        this.triangleBuffer = this.createGPUBuffer(this.mesh.triangleIndices, usageFlags);
        this.wireframeBuffer = this.createGPUBuffer(this.mesh.wireframeIndices, usageFlags);

        usageFlags = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;
        this.camera = {angle: 0., distance: 12.0, height: 10.};
        let cameraParams = new Float32Array([this.camera.angle, this.camera.distance, this.camera.height]);
        this.camBuffer = this.createGPUBuffer(cameraParams, usageFlags);
        this.colorGroupSizeBuffer = this.createGPUBuffer(this.mesh.colorGroupSizes, usageFlags);
        this.colorGroupOffsetBuffer = this.createGPUBuffer(this.mesh.colorGroupOffsets, usageFlags);
        
        let meshDim = new Uint32Array([this.mesh.nodesNumX, this.mesh.nodesNumY]);
        let sParam = new Float32Array([
            this.simParam.dt, this.mesh.dx, this.mesh.dy, 
            this.simParam.elasticK, this.simParam.gravityG, 
            this.simParam.omega, this.simParam.amplitude
        ]);    
        this.simBuffer = this.device.createBuffer({
            size: meshDim.byteLength + sParam.byteLength,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(this.simBuffer, 0, meshDim);
        this.device.queue.writeBuffer(this.simBuffer, meshDim.byteLength, sParam);

        await this.initRenderShader();
        await this.initComputeShader();
        this.colorFlag = false;

        this.render(this.colorGroupSizeBuffer,  this.colorGroupOffsetBuffer);
        this.animFlag = false;
        document.getElementById('runBtn').disabled = false;
    }

    freeGPU() {
        let buffers = [this.positionBuffer, this.newPosBuffer, this.velocityBuffer, this.invMassBuffer, 
                    this.triangleBuffer, this.wireframeBuffer, this.camBuffer, this.simBuffer, this.phaseBuffer ];
        buffers.forEach(buffer => {
            if(buffer) {
                buffer.destroy();
            }
        });    
        if(this.device) {
            this.device.destroy();
            this.device = null;
        }
    }
}

// Initialization
let model = new Model();
initParams = {
    lengthX: 10., lengthY: 10., 
    cellsX: 25, cellsY: 25, dt: 0.01,
    elasticK: 0.5, gravityG: 0.0,
    omega: 5.0, amplitude: 0.5
};

model.init(initParams);

for (const [key, value] of Object.entries(initParams)) {
  document.getElementById(key).value = value;
}

// Set parameters from form
document.getElementById('params').addEventListener('submit', function(e) {
    e.preventDefault(); 

    const params = {};
    const elements = this.elements;
    for (let element of elements) {
        if (element.name) {
            if (element.type === 'number') {
                params[element.name] = element.valueAsNumber;
            } 
            else if(element.type === 'checkbox') {
                params[element.name] = element.checked;                
            }
            else {
                params[element.name] = element.value;
            }
        }
    }
    model.freeGPU();
    if(!params['gravityFlag']) {
        params['gravityG'] = 0.0;
    }
    if(!params['oscillationFlag']) {
        params['omega'] = -1.0;
    }
    model.init(params);
});

// Turn on/off gravity
document.getElementById('gravityFlag').addEventListener('change', function() {
    document.getElementById('gravityG').disabled = !this.checked;
});

document.getElementById('colorFlag').addEventListener('change', function() {
    model.colorFlag = this.checked;
    model.render();
});

// Turn on/off oscillations
document.getElementById('oscillationFlag').addEventListener('change', function() {
    document.getElementById('omega').disabled = !this.checked;
    document.getElementById('amplitude').disabled = !this.checked;
});

// Animate
async function anim() {
    await model.compute();
    await model.render();
    if(model.animFlag) {
        requestAnimationFrame(anim);
    }
}

document.getElementById('run').addEventListener('submit', function(e) {
    e.preventDefault();
    if(model.animFlag) {
        model.animFlag = false;
        document.getElementById('runBtn').textContent = '▶ Run';
    }
    else {       
        model.animFlag = true;
        document.getElementById('runBtn').textContent = '⏹ Stop';
        requestAnimationFrame(anim);
    }
});

// Mouse camera control
let isDragging = false;
let previousMouse = { x: 0, y: 0 };

// Rotation
canvas.addEventListener('mousedown', (e) => {
    isDragging = true;
    previousMouse.x = e.clientX;
    previousMouse.y = e.clientY;
    canvas.style.cursor = 'grabbing';
});
canvas.addEventListener('mousemove', (e) => {
    if (!isDragging) return;
    
    const deltaX = e.clientX - previousMouse.x;
    const deltaY = e.clientY - previousMouse.y;
    
    // horizontal
    model.camera.angle -= deltaX * 0.01;
    
    // vertical
    model.camera.height = Math.max(-30.0, Math.min(30.0, model.camera.height + deltaY * 0.05));
    
    previousMouse.x = e.clientX;
    previousMouse.y = e.clientY;
    model.updateCameraParams();
});

canvas.addEventListener('mouseup', () => {
    isDragging = false;
    canvas.style.cursor = 'grab';
});

canvas.addEventListener('mouseleave', () => {
    isDragging = false;
    canvas.style.cursor = 'grab';
});
// Zoom
canvas.addEventListener('wheel', (e) => {
    e.preventDefault();
    model.camera.distance = Math.max(-30.0, Math.min(30.0, 
        model.camera.distance + e.deltaY * 0.005
    ));    
    model.updateCameraParams();
}, { passive: false });