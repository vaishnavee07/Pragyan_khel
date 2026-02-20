content = open('frontend_v2.html').read()
checks = [
    ('Cinematic Autofocus button',  'Cinematic Autofocus' in content),
    ('data-mode=autofocus',         'data-mode="autofocus"' in content),
    ('autofocus_click send',        'autofocus_click' in content),
    ('autofocus_double_click send', 'autofocus_double_click' in content),
    ('JSON mode switch',            "type: 'switch_mode'" in content),
    ('frame guard (if data.frame)', 'if (data.frame)' in content),
    ('data.type early return',      'if (data.type)' in content),
    ('console.log click coords',    'frame coords' in content),
    ('focus ring overlay',          'focusRing' in content),
    ('focus dot overlay',           'focusDot' in content),
    ('autofocus hint overlay',      'autofocusHint' in content),
]
all_ok = True
for label, result in checks:
    status = '[OK]  ' if result else '[FAIL]'
    if not result:
        all_ok = False
    print(f'{status} {label}')
print()
print('RESULT:', 'ALL CHECKS PASSED' if all_ok else 'SOME CHECKS FAILED')
