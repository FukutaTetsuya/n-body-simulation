#include <memory>
#include<string>

namespace SimulatorBase{

using FloatArray1D = std::unique_ptr<float[]>;
using FloatArray3D = std::unique_ptr<float[]>[3];

class NBodySimulator {
protected:
    virtual void initialize(int N, float dt, std::string filename) = 0;
    virtual void evolve_single_step() = 0;
    virtual void show_total_energy() const = 0;
    virtual void dump_coordinate(int step, std::string first_or_last_item) const = 0;
    virtual void ending() = 0;
    
    int N;
    float dt;
    std::string coordinate_file_name;
    FloatArray1D mass;
    FloatArray3D r;
    FloatArray3D v;
    FloatArray3D a;
};
}