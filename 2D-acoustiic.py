import numpy as np
import matplotlib.pyplot as plt

# 1. Define computational grid

nx = 401
nz = 151

dx = 20.0
dz = 10.0

# 2. Create spatial coordinates

x = np.arange(nx) * dx
z = np.arange(nz) * dz

# 3. Print model dimensions

print("Model width:", x[-1], "m")
print("Model depth:", z[-1], "m")

print("Number of grid points:", nx * nz)

# Create a homogeneous velocity model

velocity = np.ones((nz, nx)) * 3000.0

# Plot velocity model

plt.figure(figsize=(11, 5))

plt.imshow(
    velocity,
    extent=[0, x[-1], z[-1], 0],
    aspect='auto',
    cmap='viridis'
)

plt.colorbar(label='Velocity (m/s)')

plt.xlabel('Distance (m)')
plt.ylabel('Depth (m)')

plt.title('Homogeneous Seismic Velocity Model')

plt.show()

def ricker_wavelet(frequency, dt, duration):

    # Time vector
    t = np.arange(0, duration, dt)

    # Time shift
    t0 = 1.5 / frequency

    # Ricker wavelet equation
    wavelet = (
        1 - 2 * (np.pi**2) * (frequency**2) * (t - t0)**2
    ) * np.exp(
        -(np.pi**2) * (frequency**2) * (t - t0)**2
    )

    return t, wavelet

# Define source parameters

frequency = 20       # Dominant frequency (Hz)
dt = 0.001            # Time sampling interval (seconds)
duration = 0.2       # Total duration (seconds)

# Generate wavelet

t, wavelet = ricker_wavelet(frequency, dt, duration)

# Plot wavelet

plt.figure(figsize=(10, 4))

plt.plot(t, wavelet, linewidth=2)

plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("20 Hz Ricker Wavelet")

plt.grid(True)
plt.axhline(0, color='black', linewidth=0.8)

plt.show()

frequencies = [10, 20, 30, 40]

plt.figure(figsize=(11, 5))

for f in frequencies:

    t, w = ricker_wavelet(f, 0.001, 0.3)

    plt.plot(t, w, label=f"{f} Hz")

plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

plt.title("Comparison of Ricker Wavelets")

plt.legend()
plt.grid()

plt.show()

velocity_value = 2000

for f in frequencies:

    wavelength = velocity_value / f

    print(
        f"Frequency: {f} Hz | "
        f"Wavelength: {wavelength:.2f} m"
    )

# Source position in metres

source_x = 1500
source_z = 20

# Convert physical coordinates into grid indices

source_ix = int(source_x / dx)
source_iz = int(source_z / dz)

print("Source horizontal index:", source_ix)
print("Source vertical index:", source_iz)

# Maximum velocity in the model
v_max = np.max(velocity)

# Calculate theoretical maximum time step
dt_max = 1 / (
    v_max * np.sqrt(1/dx**2 + 1/dz**2)
)

# Choose a safety factor
safety_factor = 0.5

# Actual simulation time step
dt = safety_factor * dt_max

print("Maximum stable time step:", dt_max, "seconds")
print("Selected time step:", dt, "seconds")

cfl = v_max * dt * np.sqrt(
    1/dx**2 + 1/dz**2
)

print("CFL number:", cfl)

if cfl <= 1:
    print("Simulation is theoretically stable.")
else:
    print("Simulation may be unstable.")

# Total simulation duration
total_time = 1.5

# Number of time steps
nt = int(total_time / dt) + 1

# Time vector
time = np.arange(nt) * dt

print("Total time:", time[-1], "seconds")
print("Number of time steps:", nt)
print("Time sampling:", dt, "seconds")

def calculate_laplacian(p, dx, dz):

    laplacian = np.zeros_like(p)

    # Second derivative in x direction
    d2p_dx2 = (
        p[1:-1, 2:]
        - 2 * p[1:-1, 1:-1]
        + p[1:-1, :-2]
    ) / dx**2

    # Second derivative in z direction
    d2p_dz2 = (
        p[2:, 1:-1]
        - 2 * p[1:-1, 1:-1]
        + p[:-2, 1:-1]
    ) / dz**2

    # Combine both derivatives
    laplacian[1:-1, 1:-1] = (
        d2p_dx2 + d2p_dz2
    )

    return laplacian

def acoustic_solver(
    velocity,
    dx,
    dz,
    dt,
    nt,
    source_ix,
    source_iz,
    wavelet
):

    # Grid dimensions
    nz, nx = velocity.shape

    # Initialise three wavefields
    p_previous = np.zeros((nz, nx))
    p_current = np.zeros((nz, nx))
    p_next = np.zeros((nz, nx))

    # Velocity squared
    velocity_squared = velocity**2

    # Store snapshots
    snapshots = []

    # Time stepping loop
    for it in range(nt):

        # Calculate spatial Laplacian
        laplacian = calculate_laplacian(
            p_current, dx, dz
        )

        # Update wavefield
        p_next[1:-1, 1:-1] = (
            2 * p_current[1:-1, 1:-1]
            - p_previous[1:-1, 1:-1]
            + velocity_squared[1:-1, 1:-1]
            * dt**2
            * laplacian[1:-1, 1:-1]
        )

        # Inject seismic source
        p_next[source_iz, source_ix] += dt**2 * wavelet[it]

        # Save snapshots every 100 time steps
        if it % 100 == 0:
            snapshots.append(p_next.copy())

        # Rotate wavefields
        p_previous = p_current.copy()
        p_current = p_next.copy()
        p_next.fill(0)

    return snapshots

# Simulation parameters
frequency = 10
dt = 0.001
total_time = 1.5

nt = int(total_time / dt) + 1

# Generate source wavelet for entire simulation
time = np.arange(nt) * dt

t0 = 1.5 / frequency

wavelet = (
    1 - 2 * (np.pi * frequency * (time - t0))**2
) * np.exp(
    -(np.pi * frequency * (time - t0))**2
)

# Run simulation

snapshots = acoustic_solver(
    velocity=velocity,
    dx=dx,
    dz=dz,
    dt=dt,
    nt=nt,
    source_ix=source_ix,
    source_iz=source_iz,
    wavelet=wavelet
)

print("Simulation completed!")
print("Number of snapshots:", len(snapshots))

# Select snapshots to visualise
selected = [1, 2, 3, 4, 5, 6, 7]

# Common colour scale
vmax = max(
    np.max(np.abs(snapshots[i]))
    for i in selected
)

fig, axes = plt.subplots(
    2, 4,
    figsize=(16, 8)
)

# Add spacing between figures
fig.subplots_adjust(
    wspace=0.35,  # Horizontal gap
    hspace=0.45   # Vertical gap
)

for ax, idx in zip(axes.flat, selected):

    im = ax.imshow(
        snapshots[idx],
        extent=[0, x[-1], z[-1], 0],
        cmap="seismic",
        vmin=-vmax,
        vmax=vmax,
        aspect="auto"
    )

    ax.set_title(
        f"Time = {idx * 100 * dt:.2f} s"
    )

    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Depth (m)")

# Remove unused subplot
axes.flat[-1].axis("off")

fig.colorbar(
    im,
    ax=axes.ravel().tolist(),
    label="Pressure amplitude",
    shrink=0.8
)

plt.suptitle(
    "2D Acoustic Wave Propagation",
    fontsize=15
)

plt.show()