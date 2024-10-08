#!/usr/bin/env python
# -*- coding: utf-8 -*-

# try:
    # from PyQt5.QtGui import *
    # from PyQt5.QtCore import *
    # from PyQt5.QtWidgets import *
# except ImportError:
    # from PyQt4.QtGui import *
    # from PyQt4.QtCore import *
    
    
import sys
if sys.version_info.major >= 3:
    import sip
    sip.setapi('QVariant', 2)
from PyQt4.QtGui import *
from PyQt4.QtCore import *


import re
import sys
import codecs
import subprocess
import os.path

from functools import partial
from collections import defaultdict

# Add internal libs
from libs import resources
from libs.constants import *
from libs.lib import struct, newAction, newIcon, addActions, fmtShortcut
from libs.settings import Settings
from libs.shape import Shape, DEFAULT_LINE_COLOR, DEFAULT_FILL_COLOR
from libs.canvas import Canvas
from libs.zoomWidget import ZoomWidget
from libs.labelDialog import LabelDialog
from libs.colorDialog import ColorDialog
from libs.labelFile import LabelFile, LabelFileError
from libs.toolBar import ToolBar
from libs.pascal_voc_io import PascalVocReader
from libs.pascal_voc_io import XML_EXT
from libs.ustr import ustr
from libs.head_model import MayaviQWidget 

__appname__ = 'Head Attribute'


def have_qstring():
    '''p3/qt5 get rid of QString wrapper as py3 has native unicode str type'''
    return not (sys.version_info.major >= 3 or QT_VERSION_STR.startswith('5.'))


def util_qt_strlistclass():
    return QStringList if have_qstring() else list


class WindowMixin(object):

    def menu(self, title, actions=None):
        menu = self.menuBar().addMenu(title)
        if actions:
            addActions(menu, actions)
        return menu

    def toolbar(self, title, actions=None):
        toolbar = ToolBar(title)
        toolbar.setObjectName(u'%sToolBar' % title)
        # toolbar.setOrientation(Qt.Vertical)
        toolbar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        if actions:
            addActions(toolbar, actions)
        self.addToolBar(Qt.LeftToolBarArea, toolbar)
        return toolbar


# PyQt5: TypeError: unhashable type: 'QListWidgetItem'
class HashableQListWidgetItem(QListWidgetItem):

    def __init__(self, *args):
        super(HashableQListWidgetItem, self).__init__(*args)

    def __hash__(self):
        return hash(id(self))


class MainWindow(QMainWindow, WindowMixin):
    FIT_WINDOW, FIT_WIDTH, MANUAL_ZOOM = list(range(3))

    def __init__(self, defaultFilename=None, defaultPrefdefClassFile=None):
        super(MainWindow, self).__init__()
        self.setWindowTitle(__appname__)
        # Save as Pascal voc xml
        self.defaultSaveDir = None
        self.usingPascalVocFormat = True
        # For loading all image under a directory
        self.mImgList = []
        self.dirname = None
        self.labelHist = []
        self.lastOpenDir = None

        # Whether we need to save or not.
        self.dirty = False

        self._noSelectionSlot = False
        self._beginner = True
        self.screencastViewer = "firefox"
        self.screencast = "https://youtu.be/p0nR2YsCY_U"

        # Main widgets and related state.
        self.labelDialog = LabelDialog(parent=self, listItem=self.labelHist)
        # self.ageDialog = LabelDialog(parent=self, listItem=['0','10','20','30','40','50','60','70','80','90'])
        self.itemsToShapes = {}
        self.shapesToItems = {}
        self.prevLabelText = ''

        listLayout = QGridLayout()
        listLayout.setContentsMargins(0, 0, 0, 0)

        # Create a widget for using default label

        self.useDefaultLabelCheckbox = QCheckBox(u'')
        self.useDefaultLabelCheckbox.setChecked(False)
        self.defaultLabelTextLine = QLineEdit()
        # useDefaultLabelQHBoxLayout = QHBoxLayout()
        #useDefaultLabelQHBoxLayout.addWidget(self.useDefaultLabelCheckbox)
        #useDefaultLabelQHBoxLayout.addWidget(self.defaultLabelTextLine)
        
        # useDefaultLabelContainer = QWidget()
        # useDefaultLabelContainer.setLayout(useDefaultLabelQHBoxLayout)

        self.genderlabel = QLabel()
        self.genderlabel.setText('gender:')
        listLayout.addWidget(self.genderlabel, 0, 0)
        self.gendergroup = QButtonGroup()
        self.gendergroup.exclusive()
        self.genderButton = QCheckBox(u'Male/Boy')
        self.genderButton.setChecked(False)
        self.genderButton.stateChanged.connect(self.btnstate_ismale)
        self.genderButton2 = QCheckBox(u'Female/Girl')
        self.genderButton2.setChecked(False)
        self.genderButton2.stateChanged.connect(self.btnstate_isfemale)
        self.gendergroup.addButton(self.genderButton)
        self.gendergroup.addButton(self.genderButton2)
        listLayout.addWidget(self.genderButton2, 1, 0)
        listLayout.addWidget(self.genderButton, 1, 1)

        self.blurrinesslabel = QLabel()
        self.blurrinesslabel.setText('blurriness:')
        listLayout.addWidget(self.blurrinesslabel, 2, 0)
        self.blurrinessgroup = QButtonGroup()
        self.blurrinessgroup.exclusive()
        self.blurrinessButton = QCheckBox(u'No blur')
        self.blurrinessButton.setChecked(False)
        self.blurrinessButton.stateChanged.connect(self.btnstate_noblur)
        self.blurrinessButton2 = QCheckBox(u'Blur')
        self.blurrinessButton2.setChecked(False)
        self.blurrinessButton2.stateChanged.connect(self.btnstate_blur)
        self.blurrinessgroup.addButton(self.blurrinessButton)
        self.blurrinessgroup.addButton(self.blurrinessButton2)
        listLayout.addWidget(self.blurrinessButton, 3, 0)
        listLayout.addWidget(self.blurrinessButton2, 3, 1)


        self.occlusionlabel = QLabel()
        self.occlusionlabel.setText('frontal:')
        listLayout.addWidget(self.occlusionlabel, 6, 0)
        self.occlusiongroup = QButtonGroup()
        self.occlusiongroup.exclusive()
        self.occlusionButton0 = QCheckBox(u'Visible')
        self.occlusionButton0.setChecked(False)
        self.occlusionButton0.stateChanged.connect(self.btnstate_visible)
        self.occlusionButton1 = QCheckBox(u'Occlusion')
        self.occlusionButton1.setChecked(False)
        self.occlusionButton1.stateChanged.connect(self.btnstate_occlusion)
        self.occlusionButton2 = QCheckBox(u'Truncated')
        self.occlusionButton2.setChecked(False)
        self.occlusionButton2.stateChanged.connect(self.btnstate_truncated)
        self.occlusionButton3 = QCheckBox(u'Invisible')
        self.occlusionButton3.setChecked(False)
        self.occlusionButton3.stateChanged.connect(self.btnstate_invisible)
        self.occlusiongroup.addButton(self.occlusionButton0)
        self.occlusiongroup.addButton(self.occlusionButton1)
        self.occlusiongroup.addButton(self.occlusionButton2)
        self.occlusiongroup.addButton(self.occlusionButton3)
        listLayout.addWidget(self.occlusionButton0, 7, 0)
        listLayout.addWidget(self.occlusionButton1, 7, 1)
        listLayout.addWidget(self.occlusionButton2, 7, 2)
        listLayout.addWidget(self.occlusionButton3, 7, 3)

        self.yawlabel = QLabel()
        self.yawlabel.setText('yaw:(-180, 180)')
        listLayout.addWidget(self.yawlabel, 8, 0)
        # self.labelContainer1 = QLineEdit()  # QWidget(), QTextEdit(), QLineEdit()
        # self.labelContainer1.setLayout(QHBoxLayout())
        # self.labelContainer1.textChanged.connect(self.btnstate_yaw)
        self.labelContainer1 = QSpinBox()
        self.labelContainer1.setLayout(QHBoxLayout())
        self.labelContainer1.setRange(-180, 180)
        self.labelContainer1.setSingleStep(2)
        # self.labelContainer1.setSuffix(" degree")
        self.labelContainer1.valueChanged.connect(self.btnstate_yaw)
        listLayout.addWidget(self.labelContainer1, 9, 0)

        self.rolllabel = QLabel()
        # self.rolllabel.setText('roll:(-90, 90)')
        self.rolllabel.setText('roll:(-180, 180)')
        listLayout.addWidget(self.rolllabel, 8, 1)
        # self.labelContainer2 = QLineEdit()  # QWidget(), QTextEdit(), QLineEdit()
        # self.labelContainer2.setLayout(QHBoxLayout())
        # self.labelContainer2.textChanged.connect(self.btnstate_roll)
        self.labelContainer2 = QSpinBox()
        self.labelContainer2.setLayout(QHBoxLayout())
        # self.labelContainer2.setRange(-90, 90)
        self.labelContainer2.setRange(-180, 180)
        self.labelContainer2.setSingleStep(2)
        # self.labelContainer2.setSuffix(" degree")
        self.labelContainer2.valueChanged.connect(self.btnstate_roll)
        listLayout.addWidget(self.labelContainer2, 9, 1)

        self.pitchlabel = QLabel()
        # self.pitchlabel.setText('pitch:(-90, 90)')
        self.pitchlabel.setText('pitch:(-180, 180)')
        listLayout.addWidget(self.pitchlabel, 8, 2)
        # self.labelContainer3 = QLineEdit()  # QWidget(), QTextEdit(), QLineEdit()
        # self.labelContainer3.setLayout(QHBoxLayout())
        # self.labelContainer3.textChanged.connect(self.btnstate_pitch)
        self.labelContainer3 = QSpinBox()
        self.labelContainer3.setLayout(QHBoxLayout())
        # self.labelContainer3.setRange(-90, 90)
        self.labelContainer3.setRange(-180, 180)
        self.labelContainer3.setSingleStep(2)
        # self.labelContainer3.setSuffix(" degree")
        self.labelContainer3.valueChanged.connect(self.btnstate_pitch)
        listLayout.addWidget(self.labelContainer3, 9, 2)

        
        self.getHeadPoseButton = QPushButton()
        self.getHeadPoseButton.setText('Head Pose\nSnapshot')
        self.getHeadPoseButton.clicked.connect(self.btnstate_getHeadPose)
        listLayout.addWidget(self.getHeadPoseButton, 8, 3, 2, 1)


        self.head_model_mayavi = MayaviQWidget()
        # Parameters: QWidget_name, row_index, column_index, row_count, column_count
        listLayout.addWidget(self.head_model_mayavi, 10, 0, 1, 4)  
        

        self.editButton = QToolButton()
        self.editButton.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        # self.ageButton = QToolButton()
        # self.ageButton.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)

        # Add some of widgets to listLayout
        #listLayout.addWidget(self.editButton)
        # listLayout.addWidget(self.ageButton, 14, 0)
        # listLayout.addWidget(useDefaultLabelContainer,14, 1)

        # Create and add a widget for showing current label items
        self.labelList = QListWidget()
        labelListContainer = QWidget()
        labelListContainer.setLayout(listLayout)
        self.labelList.itemActivated.connect(self.labelSelectionChanged)
        self.labelList.itemSelectionChanged.connect(self.labelSelectionChanged)
        # self.labelList.itemDoubleClicked.connect(self.editAge)
        # '''Connect to itemChanged to detect checkbox changes.'''
        self.labelList.itemChanged.connect(self.labelItemChanged)
        # listLayout.addWidget(self.labelList)

        self.dock = QDockWidget(u'Attributes', self)
        self.dock.setObjectName(u'Labels')
        self.dock.setWidget(labelListContainer)

        # Tzutalin 20160906 : Add file list and dock to move faster
        self.fileListWidget = QListWidget()
        self.fileListWidget.itemDoubleClicked.connect(self.fileitemDoubleClicked)
        filelistLayout = QVBoxLayout()
        filelistLayout.setContentsMargins(0, 0, 0, 0)
        filelistLayout.addWidget(self.fileListWidget)
        fileListContainer = QWidget()
        fileListContainer.setLayout(filelistLayout)
        self.filedock = QDockWidget(u'File List', self)
        self.filedock.setObjectName(u'Files')
        self.filedock.setWidget(fileListContainer)

        self.zoomWidget = ZoomWidget()
        self.colorDialog = ColorDialog(parent=self)

        self.canvas = Canvas()
        self.canvas.zoomRequest.connect(self.zoomRequest)

        scroll = QScrollArea()
        scroll.setWidget(self.canvas)
        scroll.setWidgetResizable(True)
        self.scrollBars = {
            Qt.Vertical: scroll.verticalScrollBar(),
            Qt.Horizontal: scroll.horizontalScrollBar()
        }
        self.scrollArea = scroll
        self.canvas.scrollRequest.connect(self.scrollRequest)

        self.canvas.newShape.connect(self.newShape)
        self.canvas.shapeMoved.connect(self.setDirty)
        self.canvas.selectionChanged.connect(self.shapeSelectionChanged)
        self.canvas.drawingPolygon.connect(self.toggleDrawingSensitive)

        self.setCentralWidget(scroll)
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock)
        # Tzutalin 20160906 : Add file list and dock to move faster
        self.addDockWidget(Qt.RightDockWidgetArea, self.filedock)
        self.dockFeatures = QDockWidget.DockWidgetClosable\
            | QDockWidget.DockWidgetFloatable
        self.dock.setFeatures(self.dock.features() ^ self.dockFeatures)

        # Actions
        action = partial(newAction, self)
        quit = action('&Quit', self.close, 'Ctrl+Q', 'quit', u'Quit application')

        open = action('&Open', self.openFile, 'Ctrl+O', 'open', u'Open image or label file')

        opendir = action('&Open Dir', self.openDir, 'Ctrl+u', 'open', u'Open Dir')

        changeSavedir = action('&Change Save Dir', self.changeSavedir,
                               'Ctrl+r', 'open', u'Change default saved Annotation dir')

        openAnnotation = action('&Open Annotation', self.openAnnotation,
                                'Ctrl+Shift+O', 'open', u'Open Annotation')

        openNextImg = action('&Next Image', self.openNextImg, 'd', 'next', u'Open Next')

        openPrevImg = action('&Prev Image', self.openPrevImg, 'a', 'prev', u'Open Prev')

        verify = action('&Verify Image', self.verifyImg, 'space', 'verify', u'Verify Image')

        save = action('&Save', self.saveFile, 'Ctrl+S', 'save', u'Save labels to file', enabled=False)
        
        saveAs = action('&Save As', self.saveFileAs,
                        'Ctrl+Shift+S', 'save-as', u'Save labels to a different file', enabled=False)
        close = action('&Close', self.closeFile,
                       'Ctrl+W', 'close', u'Close current file')
        color1 = action('Box &Line Color', self.chooseColor1,
                        'Ctrl+L', 'color_line', u'Choose Box line color')
        color2 = action('Box &Fill Color', self.chooseColor2,
                        'Ctrl+Shift+L', 'color', u'Choose Box fill color')

        createMode = action('Create\nRectBox', self.setCreateMode,
                            'Ctrl+N', 'new', u'Start drawing Boxs', enabled=False)
        editMode = action('&Edit\nRectBox', self.setEditMode,
                          'Ctrl+J', 'edit', u'Move and edit Boxs', enabled=False)

        create = action('Create\nRectBox', self.createShape, 'F', 'new', u'Draw a new Box', enabled=False)
        delete = action('Delete\nRectBox', self.deleteSelectedShape, 'Delete', 'delete', u'Delete', enabled=False)
        copy = action('&Duplicate\nRectBox', self.copySelectedShape,
                      'Ctrl+D', 'copy', u'Create a duplicate of the selected Box', enabled=False)

        advancedMode = action('&Advanced Mode', self.toggleAdvancedMode,
                              'Ctrl+Shift+A', 'expert', u'Switch to advanced mode', checkable=True)

        hideAll = action('&Hide\nRectBox', partial(self.togglePolygons, False),
                         'Ctrl+H', 'hide', u'Hide all Boxs',
                         enabled=False)
        showAll = action('&Show\nRectBox', partial(self.togglePolygons, True),
                         'Ctrl+A', 'hide', u'Show all Boxs',
                         enabled=False)

        help = action('&Tutorial', self.tutorial, 'Ctrl+T', 'help', u'Show demos')

        zoom = QWidgetAction(self)
        zoom.setDefaultWidget(self.zoomWidget)
        self.zoomWidget.setWhatsThis(
            u"Zoom in or out of the image. Also accessible with"
            " %s and %s from the canvas." % (fmtShortcut("Ctrl+[-+]"),
                                             fmtShortcut("Ctrl+Wheel")))
        self.zoomWidget.setEnabled(False)

        zoomIn = action('Zoom &In', partial(self.addZoom, 10),
                        'Ctrl++', 'zoom-in', u'Increase zoom level', enabled=False)
        zoomOut = action('&Zoom Out', partial(self.addZoom, -10),
                         'Ctrl+-', 'zoom-out', u'Decrease zoom level', enabled=False)
        zoomOrg = action('&Original size', partial(self.setZoom, 100),
                         'Ctrl+=', 'zoom', u'Zoom to original size', enabled=False)
        fitWindow = action('&Fit Window', self.setFitWindow,
                           'Ctrl+F', 'fit-window', u'Zoom follows window size',
                           checkable=True, enabled=False)
        fitWidth = action('Fit &Width', self.setFitWidth,
                          'Ctrl+Shift+F', 'fit-width', u'Zoom follows window width',
                          checkable=True, enabled=False)
                          
        # Group zoom controls into a list for easier toggling.
        zoomActions = (self.zoomWidget, zoomIn, zoomOut, zoomOrg, fitWindow, fitWidth)
        self.zoomMode = self.MANUAL_ZOOM
        self.scalers = {
            self.FIT_WINDOW: self.scaleFitWindow,
            self.FIT_WIDTH: self.scaleFitWidth,
            # Set to one to scale to 100% when loading files.
            self.MANUAL_ZOOM: lambda: 1,
        }

        edit = action('Edit Label', self.editLabel,
                      'Ctrl+1', 'edit', u'Modify the information of the selected Box',
                      enabled=False)
        self.editButton.setDefaultAction(edit)

        # age = action('Edit Age', self.editAge,
                      # 'Ctrl+E', 'age', u'Modify the age of the selected Box',
                      # enabled=False)
        # self.ageButton.setDefaultAction(age)

        shapeLineColor = action('Shape &Line Color', self.chshapeLineColor,
                                icon='color_line', tip=u'Change the line color for this specific shape',
                                enabled=False)
        shapeFillColor = action('Shape &Fill Color', self.chshapeFillColor,
                                icon='color', tip=u'Change the fill color for this specific shape',
                                enabled=False)

        labels = self.dock.toggleViewAction()
        labels.setText('Show/Hide Label Panel')
        labels.setShortcut('Ctrl+Shift+L')

        # Lavel list context menu.
        labelMenu = QMenu()
        # addActions(labelMenu, (edit, age, delete))
        addActions(labelMenu, (edit, delete))
        self.labelList.setContextMenuPolicy(Qt.CustomContextMenu)
        self.labelList.customContextMenuRequested.connect(self.popLabelListMenu)

        # Store actions for further handling.
        self.actions = struct(save=save, saveAs=saveAs, open=open, close=close,
                              lineColor=color1, fillColor=color2,
                              create=create, delete=delete, edit=edit, copy=copy,
                              createMode=createMode, editMode=editMode, advancedMode=advancedMode,
                              shapeLineColor=shapeLineColor, shapeFillColor=shapeFillColor,
                              zoom=zoom, zoomIn=zoomIn, zoomOut=zoomOut, zoomOrg=zoomOrg,
                              fitWindow=fitWindow, fitWidth=fitWidth,
                              zoomActions=zoomActions,
                              fileMenuActions=(open, opendir, save, saveAs, close, quit),
                              beginner=(), advanced=(),
                              editMenu=(edit, copy, delete, None, color1, color2),
                              beginnerContext=(create, edit, copy, delete),
                              advancedContext=(createMode, editMode, edit, copy,
                                               delete, shapeLineColor, shapeFillColor),
                              onLoadActive=(close, create, createMode, editMode),
                              onShapesPresent=(saveAs, hideAll, showAll),
                              # age = age
                              )

        self.menus = struct(
            file=self.menu('&File'),
            edit=self.menu('&Edit'),
            view=self.menu('&View'),
            help=self.menu('&Help'),
            recentFiles=QMenu('Open &Recent'),
            labelList=labelMenu)

        # Auto saving : Enble auto saving if pressing next
        self.autoSaving = QAction("Auto Saving", self)
        self.autoSaving.setCheckable(True)

        # Sync single class mode from PR#106
        self.singleClassMode = QAction("Single Class Mode", self)
        self.singleClassMode.setShortcut("Ctrl+Shift+S")
        self.singleClassMode.setCheckable(True)
        self.lastLabel = None

        addActions(self.menus.file,
                   (open, opendir, changeSavedir, openAnnotation, self.menus.recentFiles, save, saveAs, close, None, quit))
        addActions(self.menus.help, (help,))
        addActions(self.menus.view, (
            self.autoSaving,
            self.singleClassMode,
            labels, advancedMode, None,
            hideAll, showAll, None,
            zoomIn, zoomOut, zoomOrg, None,
            fitWindow, fitWidth))

        self.menus.file.aboutToShow.connect(self.updateFileMenu)

        # Custom context menu for the canvas widget:
        addActions(self.canvas.menus[0], self.actions.beginnerContext)
        addActions(self.canvas.menus[1], (
            action('&Copy here', self.copyShape),
            action('&Move here', self.moveShape)))

        self.tools = self.toolbar('Tools')
        self.actions.beginner = (
            open, opendir, changeSavedir, openNextImg, openPrevImg, verify, save, None, create, copy, delete, None,
            zoomIn, zoom, zoomOut, fitWindow, fitWidth)

        self.actions.advanced = (
            open, opendir, changeSavedir, openNextImg, openPrevImg, save, None,
            createMode, editMode, None,
            hideAll, showAll)

        self.statusBar().showMessage('%s started.' % __appname__)
        self.statusBar().show()

        # Application state.
        self.image = QImage()
        self.filePath = ustr(defaultFilename)
        self.recentFiles = []
        self.maxRecent = 7
        self.lineColor = None
        self.fillColor = None
        self.zoom_level = 100
        self.fit_window = False
        
        # Add Chris
        self.ismale = False
        self.isfemale = False

        self.blur = False
        self.noblur = False

        # self.emo_normal = False
        # self.emo_positive = False
        # self.emo_negative = False
        # self.emo_unknown = False

        self.face_visible = False
        self.face_occlusion = False
        self.face_truncated = False
        self.face_invisible = False

        self.yaw = 0
        self.roll = 0
        self.pitch = 0


        # Load predefined classes to the list
        self.loadPredefinedClasses(defaultPrefdefClassFile)

        self.settings = Settings()
        self.settings.load()
        settings = self.settings

        ## Fix the compatible issue for qt4 and qt5. Convert the QStringList to python list
        if settings.get(SETTING_RECENT_FILES):
            if have_qstring():
                recentFileQStringList = settings.get(SETTING_RECENT_FILES)
                self.recentFiles = [ustr(i) for i in recentFileQStringList]
            else:
                self.recentFiles = recentFileQStringList = settings.get(SETTING_RECENT_FILES)

        size = settings.get(SETTING_WIN_SIZE, QSize(600, 500))
        position = settings.get(SETTING_WIN_POSE, QPoint(0, 0))
        self.resize(size)
        self.move(position)
        saveDir = ustr(settings.get(SETTING_SAVE_DIR, None))
        self.lastOpenDir = ustr(settings.get(SETTING_LAST_OPEN_DIR, None))
        if saveDir is not None and os.path.exists(saveDir):
            self.defaultSaveDir = saveDir
            self.statusBar().showMessage('%s started. Annotation will be saved to %s' %
                                         (__appname__, self.defaultSaveDir))
            self.statusBar().show()

        # or simply:
        # self.restoreGeometry(settings[SETTING_WIN_GEOMETRY]
        self.restoreState(settings.get(SETTING_WIN_STATE, QByteArray()))
        self.lineColor = QColor(settings.get(SETTING_LINE_COLOR, Shape.line_color))
        self.fillColor = QColor(settings.get(SETTING_FILL_COLOR, Shape.fill_color))
        Shape.line_color = self.lineColor
        Shape.fill_color = self.fillColor

        Shape.ismale = self.ismale
        Shape.isfemale = self.isfemale

        Shape.blur = self.blur
        Shape.noblur = self.noblur
        
        # Shape.emo_normal = self.emo_normal
        # Shape.emo_positive = self.emo_positive
        # Shape.emo_negative = self.emo_negative
        # Shape.emo_unknown = self.emo_unknown

        Shape.face_visible = self.face_visible
        Shape.face_occlusion = self.face_occlusion
        Shape.face_truncated = self.face_truncated
        Shape.face_invisible = self.face_invisible

        Shape.yaw = self.yaw
        Shape.roll = self.roll
        Shape.pitch = self.pitch

        def xbool(x):
            if isinstance(x, QVariant):
                return x.toBool()
            return bool(x)

        if xbool(settings.get(SETTING_ADVANCE_MODE, False)):
            self.actions.advancedMode.setChecked(True)
            self.toggleAdvancedMode()

        # Populate the File menu dynamically.
        self.updateFileMenu()
        
        # Since loading the file may take some time, make sure it runs in the background.
        self.queueEvent(partial(self.loadFile, self.filePath or ""))

        # Callbacks:
        self.zoomWidget.valueChanged.connect(self.paintCanvas)

        self.populateModeActions()

    ## Support Functions ##

    def noShapes(self):
        return not self.itemsToShapes

    def toggleAdvancedMode(self, value=True):
        self._beginner = not value
        self.canvas.setEditing(True)
        self.populateModeActions()
        self.editButton.setVisible(not value)
        if value:
            self.actions.createMode.setEnabled(True)
            self.actions.editMode.setEnabled(False)
            self.dock.setFeatures(self.dock.features() | self.dockFeatures)
        else:
            self.dock.setFeatures(self.dock.features() ^ self.dockFeatures)

    def populateModeActions(self):
        if self.beginner():
            tool, menu = self.actions.beginner, self.actions.beginnerContext
        else:
            tool, menu = self.actions.advanced, self.actions.advancedContext
        self.tools.clear()
        addActions(self.tools, tool)
        self.canvas.menus[0].clear()
        addActions(self.canvas.menus[0], menu)
        self.menus.edit.clear()
        actions = (self.actions.create,) if self.beginner()\
            else (self.actions.createMode, self.actions.editMode)
        addActions(self.menus.edit, actions + self.actions.editMenu)

    def setBeginner(self):
        self.tools.clear()
        addActions(self.tools, self.actions.beginner)

    def setAdvanced(self):
        self.tools.clear()
        addActions(self.tools, self.actions.advanced)

    def setDirty(self):
        self.dirty = True
        self.actions.save.setEnabled(True)

    def setClean(self):
        self.dirty = False
        self.actions.save.setEnabled(False)
        self.actions.create.setEnabled(True)

    def toggleActions(self, value=True):
        """Enable/Disable widgets which depend on an opened image."""
        for z in self.actions.zoomActions:
            z.setEnabled(value)
        for action in self.actions.onLoadActive:
            action.setEnabled(value)

    def queueEvent(self, function):
        QTimer.singleShot(0, function)

    def status(self, message, delay=5000):
        self.statusBar().showMessage(message, delay)

    def resetState(self):
        self.itemsToShapes.clear()
        self.shapesToItems.clear()
        self.labelList.clear()
        self.filePath = None
        self.imageData = None
        self.labelFile = None
        self.canvas.resetState()

    def currentItem(self):
        items = self.labelList.selectedItems()
        if items:
            return items[0]
        return None

    def addRecentFile(self, filePath):
        if filePath in self.recentFiles:
            self.recentFiles.remove(filePath)
        elif len(self.recentFiles) >= self.maxRecent:
            self.recentFiles.pop()
        self.recentFiles.insert(0, filePath)

    def beginner(self):
        return self._beginner

    def advanced(self):
        return not self.beginner()

    ## Callbacks ##
    def tutorial(self):
        subprocess.Popen([self.screencastViewer, self.screencast])

    def createShape(self):
        assert self.beginner()
        self.canvas.setEditing(False)
        self.actions.create.setEnabled(False)

    def toggleDrawingSensitive(self, drawing=True):
        """In the middle of drawing, toggling between modes should be disabled."""
        self.actions.editMode.setEnabled(not drawing)
        if not drawing and self.beginner():
            # Cancel creation.
            print('Cancel creation.')
            self.canvas.setEditing(True)
            self.canvas.restoreCursor()
            self.actions.create.setEnabled(True)

    def toggleDrawMode(self, edit=True):
        self.canvas.setEditing(edit)
        self.actions.createMode.setEnabled(edit)
        self.actions.editMode.setEnabled(not edit)

    def setCreateMode(self):
        assert self.advanced()
        self.toggleDrawMode(False)

    def setEditMode(self):
        assert self.advanced()
        self.toggleDrawMode(True)
        self.labelSelectionChanged()

    def updateFileMenu(self):
        currFilePath = self.filePath

        def exists(filename):
            return os.path.exists(filename)
        menu = self.menus.recentFiles
        menu.clear()
        files = [f for f in self.recentFiles if f !=
                 currFilePath and exists(f)]
        for i, f in enumerate(files):
            icon = newIcon('labels')
            action = QAction(
                icon, '&%d %s' % (i + 1, QFileInfo(f).fileName()), self)
            action.triggered.connect(partial(self.loadRecent, f))
            menu.addAction(action)

    def popLabelListMenu(self, point):
        self.menus.labelList.exec_(self.labelList.mapToGlobal(point))

    def editLabel(self, item=None):
        if not self.canvas.editing():
            return
        item = item if item else self.currentItem()
        text = self.labelDialog.popUp(item.text())
        if text is not None:
            item.setText(text)
            self.setDirty()

    # def editAge(self, item=None):
        # if not self.canvas.editing():
            # return
        # item = item if item else self.currentItem()
        # text = self.ageDialog.popUp(item.text())
        # if text is not None:
            # item.setText(text)
            # self.setDirty()

    # Tzutalin 20160906 : Add file list and dock to move faster
    def fileitemDoubleClicked(self, item=None):
        currIndex = self.mImgList.index(ustr(item.text()))
        if currIndex < len(self.mImgList):
            filename = self.mImgList[currIndex]
            if filename:
                self.loadFile(filename)

    # React to canvas signals.
    def shapeSelectionChanged(self, selected=False):
        if self._noSelectionSlot:
            self._noSelectionSlot = False
        else:
            shape = self.canvas.selectedShape
            if shape:
                self.shapesToItems[shape].setSelected(True)
            else:
                self.labelList.clearSelection()
        self.actions.delete.setEnabled(selected)
        self.actions.copy.setEnabled(selected)
        self.actions.edit.setEnabled(selected)
        self.actions.shapeLineColor.setEnabled(selected)
        self.actions.shapeFillColor.setEnabled(selected)

    def addLabel(self, shape):
        item = HashableQListWidgetItem(shape.label)
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(Qt.Checked)
        self.itemsToShapes[item] = shape
        self.shapesToItems[shape] = item
        # item.setText(str(shape.age))
        item.setText("head")
        self.labelList.addItem(item)
        for action in self.actions.onShapesPresent:
            action.setEnabled(True)

    def remLabel(self, shape):
        if shape is None:
            # print('rm empty label')
            return
        item = self.shapesToItems[shape]
        self.labelList.takeItem(self.labelList.row(item))
        del self.shapesToItems[shape]
        del self.itemsToShapes[item]

    def loadLabels(self, shapes):
        s = []
        for label, points, line_color, fill_color,\
            ismale, isfemale, noblur, blur,\
            face_visible, face_occlusion, face_truncated, face_invisible,\
            yaw, roll, pitch in shapes:
            # emo_normal, emo_positive, emo_negative, emo_unknown,\
            # face_visible, face_occlusion, face_truncated, face_invisible,\
            # yaw, roll, pitch in shapes:
            shape = Shape(label=label,
                          ismale=ismale, isfemale=isfemale,
                          noblur=noblur, blur=blur,
                          # emo_normal=emo_normal, emo_positive=emo_positive, 
                          # emo_negative=emo_negative, emo_unknown=emo_unknown,
                          face_visible=face_visible, face_occlusion=face_occlusion,
                          face_truncated=face_truncated, face_invisible=face_invisible,
                          yaw=yaw, roll=roll, pitch=pitch)
            for x, y in points:
                shape.addPoint(QPointF(x, y))
            shape.close()
            s.append(shape)
            self.addLabel(shape)
            
            # for QLineEdit()
            # self.labelContainer1.setText(str(shape.yaw))
            # self.labelContainer2.setText(str(shape.roll))
            # self.labelContainer3.setText(str(shape.pitch))
            
            # for QSpinBox()
            self.labelContainer1.setValue(shape.yaw)
            self.labelContainer2.setValue(shape.roll)
            self.labelContainer3.setValue(shape.pitch)
            
            if line_color:
                shape.line_color = QColor(*line_color)
            if fill_color:
                shape.fill_color = QColor(*fill_color)

        self.canvas.loadShapes(s)

    def saveLabels(self, annotationFilePath):
        annotationFilePath = ustr(annotationFilePath)
        if self.labelFile is None:
            self.labelFile = LabelFile()
            self.labelFile.verified = self.canvas.verified

        def format_shape(s):
            return dict(
                label=s.label,
                line_color=s.line_color.getRgb() if s.line_color != self.lineColor else None,
                fill_color=s.fill_color.getRgb() if s.fill_color != self.fillColor else None,
                points=[(p.x(), p.y()) for p in s.points],
                ismale = s.ismale, isfemale = s.isfemale, noblur=s.noblur, blur=s.blur,
                # emo_normal=s.emo_normal, emo_positive=s.emo_positive, 
                # emo_negative=s.emo_negative, emo_unknown=s.emo_unknown,
                face_visible=s.face_visible, face_occlusion=s.face_occlusion,
                face_truncated=s.face_truncated, face_invisible=s.face_invisible,
                yaw=s.yaw, roll=s.roll, pitch=s.pitch)

        shapes = [format_shape(shape) for shape in self.canvas.shapes]
        # Can add differrent annotation formats here
        try:
            if self.usingPascalVocFormat is True:
                # This writing format has bug when building with pyinstaller "LookupError: unknown encoding: cp65001"
                print('Img: ' + self.filePath + ' --> \n Its xml: ' + annotationFilePath)
                # print ('Img: ' + self.filePath.encode('utf-8') + ' --> Its xml: ' + annotationFilePath.encode('utf-8'))
                self.labelFile.savePascalVocFormat(annotationFilePath, shapes, self.filePath, self.imageData,
                                                   self.lineColor.getRgb(), self.fillColor.getRgb())
            else:
                self.labelFile.save(annotationFilePath, shapes, self.filePath, self.imageData,
                                    self.lineColor.getRgb(), self.fillColor.getRgb())
            return True
        except LabelFileError as e:
            self.errorMessage(u'Error saving label data', u'<b>%s</b>' % e)
            return False

    def copySelectedShape(self):
        self.addLabel(self.canvas.copySelectedShape())
        # fix copy and delete
        self.shapeSelectionChanged(True)

    def printthisshape(self, shape):

        total_str = "Gender:"
        if shape.isfemale:
            total_str += " female 0, "
        elif shape.ismale:
            total_str += " male 1, "

        total_str += "Blurriness:"
        if shape.noblur:
            total_str += " no blur 0, "
        elif shape.blur:
            total_str += " blur 1, "

        total_str += "Frontal:"
        if shape.face_visible:
            total_str += " visible 0, "
        elif shape.face_occlusion:
            total_str += " occlusion 1, "
        elif shape.face_truncated:
            total_str += " truncated 2, "
        elif shape.face_invisible:
            total_str += " invisible 3, "

        total_str += "yaw:"+str(shape.yaw)
        total_str += ",roll:"+str(shape.roll)
        total_str += ",pitch:"+str(shape.pitch)

        print(total_str)


    def labelSelectionChanged(self):
        item = self.currentItem()
        if item and self.canvas.editing():
            self._noSelectionSlot = True
            self.canvas.selectShape(self.itemsToShapes[item])
            shape = self.itemsToShapes[item]

            print("\n labelSelectionChanged:")
            self.printthisshape(shape)
            
            if shape.ismale:
                self.genderButton.setChecked(True)
            if shape.isfemale:
                self.genderButton2.setChecked(True)
                
            if shape.noblur:
                self.blurrinessButton.setChecked(True)
            if shape.blur:
                self.blurrinessButton2.setChecked(True)
            
            if shape.face_visible:
                self.occlusionButton0.setChecked(True)
            if shape.face_occlusion:
                self.occlusionButton1.setChecked(True)
            if shape.face_truncated:
                self.occlusionButton2.setChecked(True)
            if shape.face_invisible:
                self.occlusionButton3.setChecked(True)

            # for QLineEdit()
            # self.labelContainer1.setText(str(shape.yaw))
            # self.labelContainer2.setText(str(shape.roll))
            # self.labelContainer3.setText(str(shape.pitch))
            
            # for QSpinBox()
            self.labelContainer1.setValue(shape.yaw)
            self.labelContainer2.setValue(shape.roll)
            self.labelContainer3.setValue(shape.pitch)
            
            ''' update 3D head pose of the new selected head bbox annotation'''
            self.head_model_mayavi.update_head_pose_with_euler_angles(
                    shape.yaw, shape.roll, shape.pitch)
            
            
    def labelItemChanged(self, item):
        print("label Item Changed")
        shape = self.itemsToShapes[item]
        label = item.text()
        if label != shape.label:
            shape.label = item.text()
            self.setDirty()
        else:  # User probably changed item visibility
            self.canvas.setShapeVisible(shape, item.checkState() == Qt.Checked)
        
    # Callback functions:
    def newShape(self):
        text = 'head'
        self.prevLabelText = text
        self.addLabel(self.canvas.setLastLabel(text))
        if self.beginner():  # Switch to edit mode.
            self.canvas.setEditing(True)
            self.actions.create.setEnabled(True)
        else:
            self.actions.editMode.setEnabled(True)
        self.setDirty()


    def scrollRequest(self, delta, orientation):
        units = - delta / (8 * 15)
        bar = self.scrollBars[orientation]
        bar.setValue(bar.value() + bar.singleStep() * units)

    def setZoom(self, value):
        self.actions.fitWidth.setChecked(False)
        self.actions.fitWindow.setChecked(False)
        self.zoomMode = self.MANUAL_ZOOM
        self.zoomWidget.setValue(value)

    def addZoom(self, increment=10):
        self.setZoom(self.zoomWidget.value() + increment)

    def zoomRequest(self, delta):
        # get the current scrollbar positions
        # calculate the percentages ~ coordinates
        h_bar = self.scrollBars[Qt.Horizontal]
        v_bar = self.scrollBars[Qt.Vertical]

        # get the current maximum, to know the difference after zooming
        h_bar_max = h_bar.maximum()
        v_bar_max = v_bar.maximum()

        # get the cursor position and canvas size
        # calculate the desired movement from 0 to 1
        # where 0 = move left
        #       1 = move right
        # up and down analogous
        cursor = QCursor()
        pos = cursor.pos()
        relative_pos = QWidget.mapFromGlobal(self, pos)

        cursor_x = relative_pos.x()
        cursor_y = relative_pos.y()

        w = self.scrollArea.width()
        h = self.scrollArea.height()

        # the scaling from 0 to 1 has some padding
        # you don't have to hit the very leftmost pixel for a maximum-left movement
        margin = 0.1
        move_x = (cursor_x - margin * w) / (w - 2 * margin * w)
        move_y = (cursor_y - margin * h) / (h - 2 * margin * h)

        # clamp the values from 0 to 1
        move_x = min(max(move_x, 0), 1)
        move_y = min(max(move_y, 0), 1)

        # zoom in
        units = delta / (8 * 15)
        scale = 10
        self.addZoom(scale * units)

        # get the difference in scrollbar values
        # this is how far we can move
        d_h_bar_max = h_bar.maximum() - h_bar_max
        d_v_bar_max = v_bar.maximum() - v_bar_max

        # get the new scrollbar values
        new_h_bar_value = h_bar.value() + move_x * d_h_bar_max
        new_v_bar_value = v_bar.value() + move_y * d_v_bar_max

        h_bar.setValue(new_h_bar_value)
        v_bar.setValue(new_v_bar_value)

    def setFitWindow(self, value=True):
        if value:
            self.actions.fitWidth.setChecked(False)
        self.zoomMode = self.FIT_WINDOW if value else self.MANUAL_ZOOM
        self.adjustScale()

    def setFitWidth(self, value=True):
        if value:
            self.actions.fitWindow.setChecked(False)
        self.zoomMode = self.FIT_WIDTH if value else self.MANUAL_ZOOM
        self.adjustScale()

    def togglePolygons(self, value):
        for item, shape in self.itemsToShapes.items():
            item.setCheckState(Qt.Checked if value else Qt.Unchecked)

    def loadFile(self, filePath=None):
        """Load the specified file, or the last opened file if None."""
        self.resetState()
        self.canvas.setEnabled(False)
        if filePath is None:
            filePath = self.settings.get(SETTING_FILENAME)

        unicodeFilePath = ustr(filePath)
        # Tzutalin 20160906 : Add file list and dock to move faster
        # Highlight the file item
        if unicodeFilePath and self.fileListWidget.count() > 0:
            index = self.mImgList.index(unicodeFilePath)
            fileWidgetItem = self.fileListWidget.item(index)
            fileWidgetItem.setSelected(True)

        if unicodeFilePath and os.path.exists(unicodeFilePath):
            if LabelFile.isLabelFile(unicodeFilePath):
                try:
                    self.labelFile = LabelFile(unicodeFilePath)
                except LabelFileError as e:
                    self.errorMessage(u'Error opening file pos_1',
                                      (u"<p><b>%s</b></p>"
                                       u"<p>Make sure <i>%s</i> is a valid label file.")
                                      % (e, unicodeFilePath))
                    self.status("Error reading %s" % unicodeFilePath)
                    return False
                self.imageData = self.labelFile.imageData
                self.lineColor = QColor(*self.labelFile.lineColor)
                self.fillColor = QColor(*self.labelFile.fillColor)
            else:
                # Load image: read data first and store for saving into label file.
                self.imageData = read(unicodeFilePath, None)
                self.labelFile = None
            image = QImage.fromData(self.imageData)
            if image.isNull():
                self.errorMessage(u'Error opening file pos_2',
                                  u"<p>Make sure <i>%s</i> is a valid image file." % unicodeFilePath)
                self.status("Error reading %s" % unicodeFilePath)
                return False
            self.status("Loaded %s" % os.path.basename(unicodeFilePath))
            self.image = image
            self.filePath = unicodeFilePath
            self.canvas.loadPixmap(QPixmap.fromImage(image))
            if self.labelFile:
                self.loadLabels(self.labelFile.shapes)
            self.setClean()
            self.canvas.setEnabled(True)
            self.adjustScale(initial=True)
            self.paintCanvas()
            self.addRecentFile(self.filePath)
            self.toggleActions(True)

            # Label xml file and show bound box according to its filename
            if self.usingPascalVocFormat is True:
                if self.defaultSaveDir is not None:
                    basename = os.path.basename(
                        os.path.splitext(self.filePath)[0]) + XML_EXT
                    xmlPath = os.path.join(self.defaultSaveDir, basename)
                    self.loadPascalXMLByFilename(xmlPath)
                else:
                    xmlPath = os.path.splitext(filePath)[0] + XML_EXT
                    if os.path.isfile(xmlPath):
                        self.loadPascalXMLByFilename(xmlPath)

            self.setWindowTitle(__appname__ + ' ' + filePath)

            # Default : select last item if there is at least one item
            if self.labelList.count():
                self.labelList.setCurrentItem(self.labelList.item(self.labelList.count()-1))
                self.labelList.item(self.labelList.count()-1).setSelected(True)

            self.canvas.setFocus(True)
            return True
        return False

    def resizeEvent(self, event):
        if self.canvas and not self.image.isNull()\
           and self.zoomMode != self.MANUAL_ZOOM:
            self.adjustScale()
        super(MainWindow, self).resizeEvent(event)

    def paintCanvas(self):
        assert not self.image.isNull(), "cannot paint null image"
        self.canvas.scale = 0.01 * self.zoomWidget.value()
        self.canvas.adjustSize()
        self.canvas.update()

    def adjustScale(self, initial=False):
        value = self.scalers[self.FIT_WINDOW if initial else self.zoomMode]()
        self.zoomWidget.setValue(int(100 * value))

    def scaleFitWindow(self):
        """Figure out the size of the pixmap in order to fit the main widget."""
        e = 2.0  # So that no scrollbars are generated.
        w1 = self.centralWidget().width() - e
        h1 = self.centralWidget().height() - e
        a1 = w1 / h1
        # Calculate a new scale value based on the pixmap's aspect ratio.
        w2 = self.canvas.pixmap.width() - 0.0
        h2 = self.canvas.pixmap.height() - 0.0
        a2 = w2 / h2
        return w1 / w2 if a2 >= a1 else h1 / h2

    def scaleFitWidth(self):
        # The epsilon does not seem to work too well here.
        w = self.centralWidget().width() - 2.0
        return w / self.canvas.pixmap.width()

    def closeEvent(self, event):
        if not self.mayContinue():
            event.ignore()
        settings = self.settings
        # If it loads images from dir, don't load it at the begining
        if self.dirname is None:
            settings[SETTING_FILENAME] = self.filePath if self.filePath else ''
        else:
            settings[SETTING_FILENAME] = ''

        settings[SETTING_WIN_SIZE] = self.size()
        settings[SETTING_WIN_POSE] = self.pos()
        settings[SETTING_WIN_STATE] = self.saveState()
        settings[SETTING_LINE_COLOR] = self.lineColor
        settings[SETTING_FILL_COLOR] = self.fillColor
        settings[SETTING_RECENT_FILES] = self.recentFiles
        settings[SETTING_ADVANCE_MODE] = not self._beginner
        if self.defaultSaveDir is not None and len(self.defaultSaveDir) > 1:
            settings[SETTING_SAVE_DIR] = ustr(self.defaultSaveDir)
        else:
            settings[SETTING_SAVE_DIR] = ""

        if self.lastOpenDir is not None and len(self.lastOpenDir) > 1:
            settings[SETTING_LAST_OPEN_DIR] = self.lastOpenDir
        else:
            settings[SETTING_LAST_OPEN_DIR] = ""

        settings.save()
        
        
    ## User Dialogs ##
    def loadRecent(self, filename):
        if self.mayContinue():
            self.loadFile(filename)

    def scanAllImages(self, folderPath):
        extensions = ['.jpeg', '.jpg', '.png', '.bmp']
        images = []

        for root, dirs, files in os.walk(folderPath):
            for file in files:
                if file.lower().endswith(tuple(extensions)):
                    relativePath = os.path.join(root, file)
                    path = ustr(os.path.abspath(relativePath))
                    images.append(path)
        images.sort(key=lambda x: x.lower())
        return images

    def changeSavedir(self, _value=False):
        if self.defaultSaveDir is not None:
            path = ustr(self.defaultSaveDir)
        else:
            path = '.'

        dirpath = ustr(QFileDialog.getExistingDirectory(self, '%s - Save to the directory' % __appname__,
            path,  QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks))

        if dirpath is not None and len(dirpath) > 1:
            self.defaultSaveDir = dirpath

        self.statusBar().showMessage('%s . Annotation will be saved to %s' %
                                     ('Change saved folder', self.defaultSaveDir))
        self.statusBar().show()

    def openAnnotation(self, _value=False):
        if self.filePath is None:
            self.statusBar().showMessage('Please select image first')
            self.statusBar().show()
            return

        path = os.path.dirname(ustr(self.filePath))\
            if self.filePath else '.'
        if self.usingPascalVocFormat:
            filters = "Open Annotation XML file (%s)" % ' '.join(['*.xml'])
            filename = ustr(QFileDialog.getOpenFileName(self,'%s - Choose a xml file' % __appname__, path, filters))
            if filename:
                if isinstance(filename, (tuple, list)):
                    filename = filename[0]
            self.loadPascalXMLByFilename(filename)

    def openDir(self, _value=False):
        if not self.mayContinue():
            return

        path = os.path.dirname(self.filePath)\
            if self.filePath else '.'

        if self.lastOpenDir is not None and len(self.lastOpenDir) > 1:
            path = self.lastOpenDir

        dirpath = ustr(QFileDialog.getExistingDirectory(self, '%s - Open Directory' % __appname__,
            path,  QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks))

        if dirpath is not None and len(dirpath) > 1:
            self.lastOpenDir = dirpath

        self.dirname = dirpath
        self.filePath = None
        self.fileListWidget.clear()
        self.mImgList = self.scanAllImages(dirpath)
        self.openNextImg()
        for imgPath in self.mImgList:
            item = QListWidgetItem(imgPath)
            self.fileListWidget.addItem(item)

    def verifyImg(self, _value=False):
        # Proceding next image without dialog if having any label
         if self.filePath is not None:
            try:
                self.labelFile.toggleVerify()
            except AttributeError:
                # If the labelling file does not exist yet, create if and
                # re-save it with the verified attribute.
                self.saveFile()
                self.labelFile.toggleVerify()

            self.canvas.verified = self.labelFile.verified
            self.paintCanvas()
            self.saveFile()

    def openPrevImg(self, _value=False):
        # Proceding prev image without dialog if having any label
        if self.autoSaving.isChecked() and self.defaultSaveDir is not None:
            if self.dirty is True:
                self.saveFile()

        if not self.mayContinue():
            return

        if len(self.mImgList) <= 0:
            return

        if self.filePath is None:
            return

        currIndex = self.mImgList.index(self.filePath)
        if currIndex - 1 >= 0:
            filename = self.mImgList[currIndex - 1]
            if filename:
                self.loadFile(filename)

    def openNextImg(self, _value=False):
        # Proceding next image without dialog if having any label
        if self.autoSaving.isChecked() and self.defaultSaveDir is not None:
            if self.dirty is True:
                self.saveFile()

        if not self.mayContinue():
            return

        if len(self.mImgList) <= 0:
            return

        filename = None
        if self.filePath is None:
            filename = self.mImgList[0]
        else:
            currIndex = self.mImgList.index(self.filePath)
            if currIndex + 1 < len(self.mImgList):
                filename = self.mImgList[currIndex + 1]

        if filename:
            self.loadFile(filename)

    def openFile(self, _value=False):
        if not self.mayContinue():
            return
        path = os.path.dirname(ustr(self.filePath)) if self.filePath else '.'
        formats = ['*.%s' % fmt.data().decode("ascii").lower() for fmt in QImageReader.supportedImageFormats()]
        filters = "Image & Label files (%s)" % ' '.join(formats + ['*%s' % LabelFile.suffix])
        filename = QFileDialog.getOpenFileName(self, '%s - Choose Image or Label file' % __appname__, path, filters)
        if filename:
            if isinstance(filename, (tuple, list)):
                filename = filename[0]
            self.loadFile(filename)

    def saveFile(self, _value=False):
        if self.defaultSaveDir is not None and len(ustr(self.defaultSaveDir)):
            if self.filePath:
                imgFileName = os.path.basename(self.filePath)
                savedFileName = os.path.splitext(imgFileName)[0] + XML_EXT
                savedPath = os.path.join(ustr(self.defaultSaveDir), savedFileName)
                self._saveFile(savedPath)
        else:
            imgFileDir = os.path.dirname(self.filePath)
            imgFileName = os.path.basename(self.filePath)
            savedFileName = os.path.splitext(imgFileName)[0] + XML_EXT
            savedPath = os.path.join(imgFileDir, savedFileName)
            self._saveFile(savedPath if self.labelFile
                           else self.saveFileDialog())

    def saveFileAs(self, _value=False):
        assert not self.image.isNull(), "cannot save empty image"
        self._saveFile(self.saveFileDialog())

    def saveFileDialog(self):
        caption = '%s - Choose File' % __appname__
        filters = 'File (*%s)' % LabelFile.suffix
        openDialogPath = self.currentPath()
        dlg = QFileDialog(self, caption, openDialogPath, filters)
        dlg.setDefaultSuffix(LabelFile.suffix[1:])
        dlg.setAcceptMode(QFileDialog.AcceptSave)
        filenameWithoutExtension = os.path.splitext(self.filePath)[0]
        dlg.selectFile(filenameWithoutExtension)
        dlg.setOption(QFileDialog.DontUseNativeDialog, False)
        if dlg.exec_():
            return dlg.selectedFiles()[0]
        return ''

    def _saveFile(self, annotationFilePath):
        if annotationFilePath and self.saveLabels(annotationFilePath):
            self.setClean()
            self.statusBar().showMessage('Saved to  %s' % annotationFilePath)
            self.statusBar().show()

    def closeFile(self, _value=False):
        if not self.mayContinue():
            return
        self.resetState()
        self.setClean()
        self.toggleActions(False)
        self.canvas.setEnabled(False)
        self.actions.saveAs.setEnabled(False)

    def mayContinue(self):
        return not (self.dirty and not self.discardChangesDialog())

    def discardChangesDialog(self):
        yes, no = QMessageBox.Yes, QMessageBox.No
        msg = u'You have unsaved changes, proceed anyway?'
        return yes == QMessageBox.warning(self, u'Attention', msg, yes | no)

    def errorMessage(self, title, message):
        return QMessageBox.critical(self, title,
                                    '<p><b>%s</b></p>%s' % (title, message))

    def currentPath(self):
        return os.path.dirname(self.filePath) if self.filePath else '.'

    def chooseColor1(self):
        color = self.colorDialog.getColor(self.lineColor, u'Choose line color',
                                          default=DEFAULT_LINE_COLOR)
        if color:
            self.lineColor = color
            # Change the color for all shape lines:
            Shape.line_color = self.lineColor
            self.canvas.update()
            self.setDirty()

    def chooseColor2(self):
        color = self.colorDialog.getColor(self.fillColor, u'Choose fill color',
                                          default=DEFAULT_FILL_COLOR)
        if color:
            self.fillColor = color
            Shape.fill_color = self.fillColor
            self.canvas.update()
            self.setDirty()

    def deleteSelectedShape(self):
        self.remLabel(self.canvas.deleteSelected())
        self.setDirty()
        if self.noShapes():
            for action in self.actions.onShapesPresent:
                action.setEnabled(False)

    def chshapeLineColor(self):
        color = self.colorDialog.getColor(self.lineColor, u'Choose line color',
                                          default=DEFAULT_LINE_COLOR)
        if color:
            self.canvas.selectedShape.line_color = color
            self.canvas.update()
            self.setDirty()

    def chshapeFillColor(self):
        color = self.colorDialog.getColor(self.fillColor, u'Choose fill color',
                                          default=DEFAULT_FILL_COLOR)
        if color:
            self.canvas.selectedShape.fill_color = color
            self.canvas.update()
            self.setDirty()

    def copyShape(self):
        self.canvas.endMove(copy=True)
        self.addLabel(self.canvas.selectedShape)
        self.setDirty()

    def moveShape(self):
        self.canvas.endMove(copy=False)
        self.setDirty()

    def loadPredefinedClasses(self, predefClassesFile):
        if os.path.exists(predefClassesFile) is True:
            with codecs.open(predefClassesFile, 'r', 'utf8') as f:
                for line in f:
                    line = line.strip()
                    if self.labelHist is None:
                        self.lablHist = [line]
                    else:
                        self.labelHist.append(line)

    def loadPascalXMLByFilename(self, xmlPath):
        if self.filePath is None:
            return
        if os.path.isfile(xmlPath) is False:
            return

        tVocParseReader = PascalVocReader(xmlPath)
        shapes = tVocParseReader.getShapes()
        self.loadLabels(shapes)
        self.canvas.verified = tVocParseReader.verified



    #button state:
    def btnstate_ismale(self, item=None):
        if not self.canvas.editing():
            return
        item = self.currentItem()
        if not item:  # If not selected Item, take the first one
            item = self.labelList.item(self.labelList.count() - 1)
        ismale = self.genderButton.isChecked()
        try:
            shape = self.itemsToShapes[item]
            if ismale != shape.ismale:
                shape.ismale = ismale
                self.setDirty()
            else:  # User probably changed item visibility
                self.canvas.setShapeVisible(shape, item.checkState() == Qt.Checked)
        except:
            pass
            
    def btnstate_isfemale(self, item=None):
        if not self.canvas.editing():
            return
        item = self.currentItem()
        if not item:  # If not selected Item, take the first one
            item = self.labelList.item(self.labelList.count() - 1)
        isfemale = self.genderButton2.isChecked()
        try:
            shape = self.itemsToShapes[item]
            if isfemale != shape.isfemale:
                shape.isfemale = isfemale
                self.setDirty()
            else:  # User probably changed item visibility
                self.canvas.setShapeVisible(shape, item.checkState() == Qt.Checked)
        except:
            pass
            
    def btnstate_noblur(self, item=None):
        if not self.canvas.editing():
            return
        item = self.currentItem()
        if not item:  # If not selected Item, take the first one
            item = self.labelList.item(self.labelList.count() - 1)
        noblur = self.blurrinessButton.isChecked()
        try:
            shape = self.itemsToShapes[item]
            if noblur != shape.noblur:
                shape.noblur = noblur
                self.setDirty()
            else:  # User probably changed item visibility
                self.canvas.setShapeVisible(shape, item.checkState() == Qt.Checked)
        except:
            pass
    def btnstate_blur(self, item=None):
        if not self.canvas.editing():
            return
        item = self.currentItem()
        if not item:  # If not selected Item, take the first one
            item = self.labelList.item(self.labelList.count() - 1)
        blur = self.blurrinessButton2.isChecked()
        try:
            shape = self.itemsToShapes[item]
            if blur != shape.blur:
                shape.blur = blur
                self.setDirty()
            else:  # User probably changed item visibility
                self.canvas.setShapeVisible(shape, item.checkState() == Qt.Checked)
        except:
            pass

    
    def btnstate_visible(self, item=None):
        if not self.canvas.editing():
            return
        item = self.currentItem()
        if not item:  # If not selected Item, take the first one
            item = self.labelList.item(self.labelList.count() - 1)
        face_visible = self.occlusionButton0.isChecked()
        try:
            shape = self.itemsToShapes[item]
            if face_visible != shape.face_visible:
                shape.face_visible = face_visible
                self.setDirty()
            else:  # User probably changed item visibility
                self.canvas.setShapeVisible(shape, item.checkState() == Qt.Checked)
        except:
            pass

    def btnstate_occlusion(self, item=None):
        if not self.canvas.editing():
            return
        item = self.currentItem()
        if not item:  # If not selected Item, take the first one
            item = self.labelList.item(self.labelList.count() - 1)
        face_occlusion = self.occlusionButton1.isChecked()
        try:
            shape = self.itemsToShapes[item]
            if face_occlusion != shape.face_occlusion:
                shape.face_occlusion = face_occlusion
                self.setDirty()
            else:  # User probably changed item visibility
                self.canvas.setShapeVisible(shape, item.checkState() == Qt.Checked)
        except:
            pass

    def btnstate_truncated(self, item=None):
        if not self.canvas.editing():
            return
        item = self.currentItem()
        if not item:  # If not selected Item, take the first one
            item = self.labelList.item(self.labelList.count() - 1)
        face_truncated = self.occlusionButton2.isChecked()
        try:
            shape = self.itemsToShapes[item]
            if face_truncated != shape.face_truncated:
                shape.face_truncated = face_truncated
                self.setDirty()
            else:  # User probably changed item visibility
                self.canvas.setShapeVisible(shape, item.checkState() == Qt.Checked)
        except:
            pass

    def btnstate_invisible(self, item=None):
        if not self.canvas.editing():
            return
        item = self.currentItem()
        if not item:  # If not selected Item, take the first one
            item = self.labelList.item(self.labelList.count() - 1)
        face_invisible = self.occlusionButton3.isChecked()
        try:
            shape = self.itemsToShapes[item]
            if face_invisible != shape.face_invisible:
                shape.face_invisible = face_invisible
                self.setDirty()
            else:  # User probably changed item visibility
                self.canvas.setShapeVisible(shape, item.checkState() == Qt.Checked)
        except:
            pass

    def btnstate_getHeadPose(self, item=None):
        if not self.canvas.editing():
            return
        item = self.currentItem()
        if not item:  # If not selected Item, take the first one
            item = self.labelList.item(self.labelList.count() - 1)
        yaw, roll, pitch = self.head_model_mayavi.calculate_euler_angles()
        yaw, roll, pitch = round(yaw), round(roll), round(pitch)
        try:
            shape = self.itemsToShapes[item]
            if yaw != shape.yaw or roll != shape.roll or pitch != shape.pitch:
                shape.yaw = yaw
                self.labelContainer1.setValue(yaw)
                shape.roll = roll
                self.labelContainer2.setValue(roll)
                shape.pitch = pitch
                self.labelContainer3.setValue(pitch)
                self.setDirty()
            else:  # User probably changed item visibility
                self.canvas.setShapeVisible(shape, item.checkState() == Qt.Checked)
        except:
            pass

    def btnstate_yaw(self, item=None):
        if not self.canvas.editing():
            return
        item = self.currentItem()
        if not item:  # If not selected Item, take the first one
            item = self.labelList.item(self.labelList.count() - 1)
        # yaw = int(self.labelContainer1.text())  # for QLineEdit()
        yaw = self.labelContainer1.value()  # for QSpinBox()
        try:
            shape = self.itemsToShapes[item]
            if yaw != shape.yaw:
                shape.yaw = yaw
                self.setDirty()
                self.head_model_mayavi.update_head_pose_with_euler_angles(
                    shape.yaw, shape.roll, shape.pitch)
            else:  # User probably changed item visibility
                self.canvas.setShapeVisible(shape, item.checkState() == Qt.Checked)
        except:
            pass

    def btnstate_roll(self, item=None):
        if not self.canvas.editing():
            return
        item = self.currentItem()
        if not item:  # If not selected Item, take the first one
            item = self.labelList.item(self.labelList.count() - 1)
        # roll = int(self.labelContainer2.text())  # for QLineEdit()
        roll = self.labelContainer2.value()  # for QSpinBox()
        try:
            shape = self.itemsToShapes[item]
            if roll != shape.roll:
                shape.roll = roll
                self.setDirty()
                self.head_model_mayavi.update_head_pose_with_euler_angles(
                    shape.yaw, shape.roll, shape.pitch)
            else:  # User probably changed item visibility
                self.canvas.setShapeVisible(shape, item.checkState() == Qt.Checked)
        except:
            pass

    def btnstate_pitch(self, item=None):
        if not self.canvas.editing():
            return
        item = self.currentItem()
        if not item:  # If not selected Item, take the first one
            item = self.labelList.item(self.labelList.count() - 1)
        # pitch = int(self.labelContainer3.text())  # for QLineEdit()
        pitch = self.labelContainer3.value()  # for QSpinBox()
        try:
            shape = self.itemsToShapes[item]
            if pitch != shape.pitch:
                shape.pitch = pitch
                self.setDirty()
                self.head_model_mayavi.update_head_pose_with_euler_angles(
                    shape.yaw, shape.roll, shape.pitch)
            else:  # User probably changed item visibility
                self.canvas.setShapeVisible(shape, item.checkState() == Qt.Checked)
        except:
            pass


def inverted(color):
    return QColor(*[255 - v for v in color.getRgb()])


def read(filename, default=None):
    try:
        with open(filename, 'rb') as f:
            return f.read()
    except:
        return default


def get_main_app(argv=[]):
    """
    Standard boilerplate Qt application code.
    Do everything but app.exec_() -- so that we can test the application in one thread
    """
    app = QApplication(argv)
    app.setApplicationName(__appname__)
    app.setWindowIcon(newIcon("app"))
    # Tzutalin 201705+: Accept extra agruments to change predefined class file
    # Usage : labelImg.py image predefClassFile
    win = MainWindow(argv[1] if len(argv) >= 2 else None,
                     argv[2] if len(argv) >= 3 else os.path.join(
                         os.path.dirname(sys.argv[0]),
                         'data', 'predefined_classes.txt'))
    win.show()
    return app, win


def main(argv=[]):
    '''construct main app and run it'''
    app, _win = get_main_app(argv)
    return app.exec_()

if __name__ == '__main__':
    sys.exit(main(sys.argv))
