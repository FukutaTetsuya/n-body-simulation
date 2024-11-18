#include <memory>

namespace SimulatorBase{

using FloatArray1D = std::unique_ptr<float[]>;
using FloatArray3D = std::unique_ptr<float[]>[3];

class NBodySimulator {
protected:
    virtual void initialize(int, float) = 0;
    //virtual void evolve_single_step() = 0;
    //virtual void show_total_energy() const = 0;
    //virtual void dump_coordinate() const = 0;
    //virtual void ending() = 0;
private:
    int N;
    float dt;
    FloatArray1D mass;
    FloatArray3D r;
    FloatArray3D v;
    FloatArray3D a;
};
}