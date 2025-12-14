import numpy as np
import torch

def calculate_power_spectrum(data_x, Lpix=3, kbins=100, dsq = False, method="torch", device="cpu"):
    #Simulation box variables
    batch, channel,*d = data_x.shape
    assert channel == 1, "Channel must be 1"
    Npix = d[-1]
    Vpix = Lpix**3
    Lbox = Npix * Lpix
    Vbox = Lbox**3

    if method == "numpy":
        kspace = np.fft.fftfreq(Npix, d=Lpix/(2*np.pi)) #Calculating wavevectors k for the simulation grid
        kx, ky, kz = np.meshgrid(kspace,kspace,kspace)
        k = np.sqrt(kx**2 + ky**2 + kz**2)
        data_k = np.fft.fftn(data_x[0,0]) #Dont need to scipy.fft.fftshift since kspace isn't fftshift'ed
        k_bin_edges = np.geomspace(np.min(k[np.nonzero(k)]), np.max(k), endpoint=True, num=kbins+1) #Bin k values and calculate power spectrum
        k_vals = np.zeros(kbins)
        P_k = np.zeros(kbins)
        for i in range(kbins):
            cond = ((k >= k_bin_edges[i]) & (k < k_bin_edges[i+1]))
            k_vals[i] = (k_bin_edges[i+1] + k_bin_edges[i])/2
            P_k[i] = (Vpix/Vbox) * Vpix * np.average(np.absolute(data_k[cond])**2)
        P_k = P_k*k_vals**3/(2*np.pi**2) if dsq else P_k
        return k_vals, P_k
    elif method == "torch":
        kspace_torch = torch.fft.fftfreq(Npix, d=Lpix/(2*np.pi), device=device)
        kx_torch, ky_torch, kz_torch = torch.meshgrid(kspace_torch,kspace_torch,kspace_torch, indexing='ij')#.view(batch,channel,*d)
        k_torch = torch.sqrt(kx_torch**2 + ky_torch**2 + kz_torch**2).to(device=device)
        data_k_torch = torch.fft.fftn(input=data_x, dim=(2,3,4)).to(device=device)
        kmin_mask_torch = k_torch > 0
        kmin_torch = torch.min(k_torch[kmin_mask_torch])
        kmax_torch = torch.max(k_torch)
        k_bin_edges_torch = torch.logspace(start=torch.log10(kmin_torch), end=torch.log10(kmax_torch), steps=kbins+1, device=device)
        k_vals_torch = torch.zeros(kbins, device=device)
        P_k_torch = torch.zeros(batch,channel,kbins, device=device)

        conditions = [(k_torch >= k_bin_edges_torch[i]) & (k_torch < k_bin_edges_torch[i+1]) for i in range(kbins)]

        for i in range(kbins):
            cond_torch = conditions[i]
            k_vals_torch[i] = (k_bin_edges_torch[i+1] + k_bin_edges_torch[i])/2
            means = torch.mean(torch.abs(data_k_torch[...,cond_torch])**2,dim=2, keepdim=False)
            P_k_torch[:,:,i] = (Vpix/Vbox) * Vpix * means
        P_k_torch = P_k_torch*k_vals_torch**3/(2*np.pi**2) if dsq else P_k_torch
        return k_vals_torch, P_k_torch
    else:
        raise ValueError("Method must be numpy or torch")
