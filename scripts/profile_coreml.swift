import CoreML
import Foundation

@available(macOS 14.4, *)
private func profileBlock(_ block: MLModelStructure.Program.Block, plan: MLComputePlan) -> [String: Int] {
    var counts: [String: Int] = [:]
    for operation in block.operations {
        if let usage = plan.deviceUsage(for: operation) {
            let device = usage.preferred.description
            counts[device, default: 0] += 1
            let supported = usage.supported.map(\.description).joined(separator: ",")
            print("\(operation.operatorName)\tpreferred=\(device)\tsupported=\(supported)")
        } else {
            counts["unknown", default: 0] += 1
            print("\(operation.operatorName)\tpreferred=unknown")
        }
        for nestedBlock in operation.blocks {
            for (device, count) in profileBlock(nestedBlock, plan: plan) {
                counts[device, default: 0] += count
            }
        }
    }
    return counts
}

@main
struct CoreMLProfiler {
    static func main() async throws {
        guard CommandLine.arguments.count == 2 else {
            FileHandle.standardError.write(Data("usage: profile_coreml MODEL\n".utf8))
            Foundation.exit(2)
        }

        let modelURL = URL(fileURLWithPath: CommandLine.arguments[1])
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .cpuAndNeuralEngine
        let plan = try await MLComputePlan.load(
            contentsOf: modelURL,
            configuration: configuration
        )

        switch plan.modelStructure {
        case .program(let program):
            var totals: [String: Int] = [:]
            for (name, function) in program.functions {
                print("function=\(name)")
                for (device, count) in profileBlock(function.block, plan: plan) {
                    totals[device, default: 0] += count
                }
            }
            print("summary")
            for (device, count) in totals.sorted(by: { $0.key < $1.key }) {
                print("\(device)\t\(count)")
            }
        case .neuralNetwork(let network):
            var totals: [String: Int] = [:]
            for layer in network.layers {
                if let usage = plan.deviceUsage(for: layer) {
                    let device = usage.preferred.description
                    totals[device, default: 0] += 1
                    let supported = usage.supported.map(\.description).joined(separator: ",")
                    print("\(layer.type)\tpreferred=\(device)\tsupported=\(supported)")
                } else {
                    totals["unknown", default: 0] += 1
                    print("\(layer.type)\tpreferred=unknown")
                }
            }
            print("summary")
            for (device, count) in totals.sorted(by: { $0.key < $1.key }) {
                print("\(device)\t\(count)")
            }
        default:
            throw NSError(
                domain: "HOMRCoreMLProfiler",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey: "Expected an MLProgram or neural network model"]
            )
        }
    }
}
