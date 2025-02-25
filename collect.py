text_s4r='''
Air RLIT[1] @ PSNR: 29.862
Air RLIT[1] @ SSIM: 0.9557
Air RLIT[1] @ LPIPS: 0.0626
Air RLIT[2] @ PSNR: 29.580
Air RLIT[2] @ SSIM: 0.9622
Air RLIT[2] @ LPIPS: 0.0516
Air Albedo @ PSNR: 25.980
Air Albedo @ SSIM: 0.9489
Air Albedo @ LPIPS: 0.0575
Air Roughness @ MSE: 0.003
Chair RLIT[1] @ PSNR: 30.260
Chair RLIT[1] @ SSIM: 0.9584
Chair RLIT[1] @ LPIPS: 0.0528
Chair RLIT[2] @ PSNR: 30.296
Chair RLIT[2] @ SSIM: 0.9553
Chair RLIT[2] @ LPIPS: 0.0532
Chair Albedo @ PSNR: 26.045
Chair Albedo @ SSIM: 0.9141
Chair Albedo @ LPIPS: 0.0701
Chair Roughness @ MSE: 0.003
Hotdog RLIT[1] @ PSNR: 28.539
Hotdog RLIT[1] @ SSIM: 0.9507
Hotdog RLIT[1] @ LPIPS: 0.0702
Hotdog RLIT[2] @ PSNR: 28.845
Hotdog RLIT[2] @ SSIM: 0.9506
Hotdog RLIT[2] @ LPIPS: 0.0740
Hotdog Albedo @ PSNR: 26.340
Hotdog Albedo @ SSIM: 0.9436
Hotdog Albedo @ LPIPS: 0.0726
Hotdog Roughness @ MSE: 0.016
Jugs RLIT[1] @ PSNR: 31.528
Jugs RLIT[1] @ SSIM: 0.9764
Jugs RLIT[1] @ LPIPS: 0.0241
Jugs RLIT[2] @ PSNR: 32.279
Jugs RLIT[2] @ SSIM: 0.9761
Jugs RLIT[2] @ LPIPS: 0.0241
Jugs Albedo @ PSNR: 28.398
Jugs Albedo @ SSIM: 0.9391
Jugs Albedo @ LPIPS: 0.0655
Jugs Roughness @ MSE: 0.002
'''


if __name__ == '__main__':

    def avg(l, fixed):
        fmt = f'.{fixed}f'
        return f'{sum(l) / len(l):{fmt}}'

    def display(l):
        print(' & '.join([str(x) for x in l]))

    lut = {}
    for key, value in [line.split(': ') for line in text_s4r.split('\n') if line]:
        lut.setdefault(key.split(' ', 1)[0], {})[key.split(' ', 1)[1]] = float(value)

    scenes = ['Air', 'Chair', 'Hotdog', 'Jugs']
    # scenes = ['Lego', 'Hotdog', 'Armadillo', 'Ficus']

    # print('NVS')
    # lst = []
    # for scene in scenes:
    #     lst.append(lut[scene]['NVS @ PSNR'])
    #     lst.append(lut[scene]['NVS @ SSIM'])
    #     lst.append(lut[scene]['NVS @ LPIPS'])
    # lst.append(avg(lst[0::3], 2))
    # lst.append(avg(lst[1::3], 4))
    # lst.append(avg(lst[2::3], 4))
    # display(lst)

    # print('Envmap6')
    # lst = []
    # for scene in scenes:
    #     lst.append(lut[scene]['RLIT[1] @ PSNR'])
    #     lst.append(lut[scene]['RLIT[1] @ SSIM'])
    #     lst.append(lut[scene]['RLIT[1] @ LPIPS'])
    # lst.append(avg(lst[0::3], 2))
    # lst.append(avg(lst[1::3], 4))
    # lst.append(avg(lst[2::3], 4))
    # display(lst)

    # print('Envmap12')
    # lst = []
    # for scene in scenes:
    #     lst.append(lut[scene]['RLIT[2] @ PSNR'])
    #     lst.append(lut[scene]['RLIT[2] @ SSIM'])
    #     lst.append(lut[scene]['RLIT[2] @ LPIPS'])
    # lst.append(avg(lst[0::3], 2))
    # lst.append(avg(lst[1::3], 4))
    # lst.append(avg(lst[2::3], 4))
    # display(lst)

    print('Relight')
    lst = []
    for scene in scenes:
        lst.append(round((lut[scene]['RLIT[1] @ PSNR'] + lut[scene]['RLIT[2] @ PSNR']) / 2, 3))
        lst.append(round((lut[scene]['RLIT[1] @ SSIM'] + lut[scene]['RLIT[2] @ SSIM']) / 2, 4))
        lst.append(round((lut[scene]['RLIT[1] @ LPIPS'] + lut[scene]['RLIT[2] @ LPIPS']) / 2, 4))
    lst.append(avg(lst[0::3], 2))
    lst.append(avg(lst[1::3], 4))
    lst.append(avg(lst[2::3], 4))
    display(lst)

    print('Albedo')
    lst = []
    for scene in scenes:
        lst.append(lut[scene]['Albedo @ PSNR'])
        lst.append(lut[scene]['Albedo @ SSIM'])
        lst.append(lut[scene]['Albedo @ LPIPS'])
    lst.append(avg(lst[0::3], 2))
    lst.append(avg(lst[1::3], 4))
    lst.append(avg(lst[2::3], 4))
    display(lst)

    print('MSE')
    lst = []
    for scene in scenes:
        lst.append(lut[scene]['Roughness @ MSE'])
    lst.append(avg(lst, 3))
    display(lst)