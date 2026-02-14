// flaretest — writes a mostly-black EXR with a single extremely bright pixel
//             in the top-right third of the image.

#include <cstdio>
#include <cstdlib>
#include <vector>

#include <OpenEXR/ImfOutputFile.h>
#include <OpenEXR/ImfChannelList.h>
#include <OpenEXR/ImfHeader.h>
#include <OpenEXR/ImfFrameBuffer.h>

int main(int argc, char *argv[])
{
    const char *out_path = (argc > 1) ? argv[1] : "flaretest.exr";

    const int W = 512;
    const int H = 256;

    // Bright pixel roughly in the top-right third
    const int bx = W * 3 / 4; // 384
    const int by = H * 1 / 4; // 128

    const float brightness = 500.0f; // extremely bright

    std::vector<float> r(W * H, 0.0f);
    std::vector<float> g(W * H, 0.0f);
    std::vector<float> b(W * H, 0.0f);

    const int idx = by * W + bx;
    r[idx] = brightness;
    g[idx] = brightness;
    b[idx] = brightness;

    printf("Writing %dx%d EXR with bright pixel at (%d, %d) = %.0f\n",
           W, H, bx, by, brightness);

    try
    {
        Imf::Header header(W, H);
        header.channels().insert("R", Imf::Channel(Imf::FLOAT));
        header.channels().insert("G", Imf::Channel(Imf::FLOAT));
        header.channels().insert("B", Imf::Channel(Imf::FLOAT));

        Imf::OutputFile file(out_path, header);

        const size_t stride = sizeof(float);
        const size_t scanline = sizeof(float) * W;

        Imf::FrameBuffer fb;
        fb.insert("R", Imf::Slice(Imf::FLOAT, (char *)r.data(), stride, scanline));
        fb.insert("G", Imf::Slice(Imf::FLOAT, (char *)g.data(), stride, scanline));
        fb.insert("B", Imf::Slice(Imf::FLOAT, (char *)b.data(), stride, scanline));

        file.setFrameBuffer(fb);
        file.writePixels(H);

        printf("Wrote: %s\n", out_path);
    }
    catch (const std::exception &e)
    {
        fprintf(stderr, "Error writing EXR: %s\n", e.what());
        return 1;
    }

    return 0;
}
