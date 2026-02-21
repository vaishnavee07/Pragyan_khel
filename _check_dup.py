import re
content = open('C:/Users/Vasihnavee/OneDrive/Desktop/SentraVision/frontend_v2.html', encoding='utf-8').read()

# Find the old injected panel (was inserted before Performance Metrics header in last session)
# Look for a second occurrence of "videoUploadPanel" id
idx1 = content.find('id="videoUploadPanel"')
idx2 = content.find('id="videoUploadPanel"', idx1 + 1)

if idx2 == -1:
    print('No duplicate videoUploadPanel found - HTML is clean')
else:
    print(f'Duplicate found at char {idx2}')
    # Show context
    print(repr(content[idx2-100:idx2+200]))
