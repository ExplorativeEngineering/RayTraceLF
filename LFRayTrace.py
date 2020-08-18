"""
LFRayTrace
"""

# Main test
import LFRayTraceVoxGenerate
import LFRayTraceVoxProjection


def main():
    print("LFRayTrace")
    uLenses = 15
    voxPitch = 26/15
    print("Generating LFRTVoxels...")
    LFRayTraceVoxGenerate.generateLFRTvoxels(uLenses, voxPitch)
    print("Projecting Samples...")
    LFRayTraceVoxProjection.projectSamples(uLenses, voxPitch)


if __name__ == "__main__":
    main()
