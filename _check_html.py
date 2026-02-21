content = open('C:/Users/Vasihnavee/OneDrive/Desktop/SentraVision/frontend_v2.html', encoding='utf-8').read()

checks = [
    ('liveModeContainer',       'id="liveModeContainer"'),
    ('videoModeContainer',      'id="videoModeContainer"'),
    ('videoUploadPanel',        'id="videoUploadPanel"'),
    ('uploadZone',              'id="uploadZone"'),
    ('vCanvas',                 'id="vCanvas"'),
    ('videoUploadModeBtn',      'id="videoUploadModeBtn"'),
    ('activateVideoUploadMode', 'activateVideoUploadMode'),
    ('exitVideoUploadMode',     'exitVideoUploadMode'),
]
ok = True
for label, needle in checks:
    count = content.count(needle)
    status = 'OK ' if count >= 1 else 'MISSING'
    dup    = '  *** DUPLICATE ***' if count > 2 else ''
    print(f'{status}  {label}: {count}x{dup}')
    if count == 0:
        ok = False
print('All present' if ok else 'MISSING ELEMENTS FOUND')
