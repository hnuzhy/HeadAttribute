# -*- mode: python ; coding: utf-8 -*-


block_cipher = None


a = Analysis(['labelImg.py'],
             pathex=[],
             binaries=[],
             datas=[
                ('C:/Users/83538/.conda/envs/py37_headpose/Lib/site-packages/numpy', 'numpy'),
                ('C:/Users/83538/.conda/envs/py37_headpose/Lib/site-packages/lxml', 'lxml'),
                ('C:/Users/83538/.conda/envs/py37_headpose/Lib/site-packages/PIL', 'PIL'),
                ('C:/Users/83538/.conda/envs/py37_headpose/Lib/site-packages/matplotlib', 'matplotlib'),
                ('C:/Users/83538/.conda/envs/py37_headpose/Lib/site-packages/vtk.py', 'vtk'),
                ('C:/Users/83538/.conda/envs/py37_headpose/Lib/site-packages/tvtk', 'tvtk'),
                ('C:/Users/83538/.conda/envs/py37_headpose/Lib/site-packages/VTK-9.1.0.dist-info', 'VTK-9.1.0.dist-info'),
                ('C:/Users/83538/.conda/envs/py37_headpose/Lib/site-packages/vtkmodules', 'vtkmodules'),
                ('C:/Users/83538/.conda/envs/py37_headpose/Lib/site-packages/_vtkmodules_static.cp37-win_amd64.pyd', '_vtkmodules_static'),
                ('C:/Users/83538/.conda/envs/py37_headpose/Lib/site-packages/mayavi', 'mayavi'),
                ('C:/Users/83538/.conda/envs/py37_headpose/Lib/site-packages/PyQt4', 'PyQt4'),
                ('C:/Users/83538/.conda/envs/py37_headpose/Lib/site-packages/pygments', 'pygments'),
                ('C:/Users/83538/.conda/envs/py37_headpose/Lib/site-packages/pyface', 'pyface'),
                ('C:/Users/83538/.conda/envs/py37_headpose/Lib/site-packages/pyface-7.3.0.dist-info', 'pyface-7.3.0.dist-info'),
                ('C:/Users/83538/.conda/envs/py37_headpose/Lib/site-packages/traits', 'traits'),
                ('C:/Users/83538/.conda/envs/py37_headpose/Lib/site-packages/traitsui', 'traitsui'),
                ('C:/Users/83538/.conda/envs/py37_headpose/Lib/site-packages/traitsui-7.2.1.dist-info', 'traitsui-7.2.1.dist-info')
             ],
             hiddenimports=[],
             hookspath=[],
             hooksconfig={},
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)

# remove warnings bboxes
for line in a.datas:
    if 'cp37-win_amd64.pyd' in line[0] or 'Qt.pyd' in line[0] or 'QtCore.pyd' in line[0] or 'QtGui.pyd' in line[0]:
        a.datas.remove(line)
             
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)

exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,  
          [],
          name='labelImg',
          debug=True,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=True,
          disable_windowed_traceback=False,
          target_arch=None,
          codesign_identity=None,
          entitlements_file=None )
