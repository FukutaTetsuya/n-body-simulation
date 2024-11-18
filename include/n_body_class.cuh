#include <memory>

namespace Simulator{

using FloatArray1D = std::unique_ptr<float[]>;
using FloatArray3D = std::unique_ptr<float[]>[3];

class NBodySimulator {
public:
    void initialize();
    void evolve_single_step();
    void show_total_energy();
    void dump_coordinate();
    void ending();
private:
    int N;
    float dt;
    FloatArray1D mass;
    FloatArray3D r;
    FloatArray3D v;
    FloatArray3D a;
};
}