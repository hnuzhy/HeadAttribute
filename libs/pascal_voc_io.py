#!/usr/bin/env python
# -*- coding: utf8 -*-
import sys
from xml.etree import ElementTree
from xml.etree.ElementTree import Element, SubElement
from lxml import etree
import codecs

XML_EXT = '.xml'
ENCODE_METHOD = 'utf-8'

class PascalVocWriter:

    def __init__(self, foldername, filename, imgSize, databaseSrc='Unknown', localImgPath=None):
        self.foldername = foldername
        self.filename = filename
        self.databaseSrc = databaseSrc
        self.imgSize = imgSize
        self.boxlist = []
        self.localImgPath = localImgPath
        self.verified = False

    def prettify(self, elem):
        """
            Return a pretty-printed XML string for the Element.
        """
        rough_string = ElementTree.tostring(elem, 'utf8')
        root = etree.fromstring(rough_string)
        return etree.tostring(root, pretty_print=True, encoding=ENCODE_METHOD).replace("  ".encode(), "\t".encode())
        # minidom does not support UTF-8
        '''reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="\t", encoding=ENCODE_METHOD)'''

    def genXML(self):
        """
            Return XML root
        """
        # Check conditions
        if self.filename is None or \
                self.foldername is None or \
                self.imgSize is None:
            return None

        top = Element('annotation')
        if self.verified:
            top.set('verified', 'yes')

        folder = SubElement(top, 'folder')
        folder.text = self.foldername

        filename = SubElement(top, 'filename')
        filename.text = self.filename

        if self.localImgPath is not None:
            localImgPath = SubElement(top, 'path')
            localImgPath.text = self.localImgPath

        source = SubElement(top, 'source')
        database = SubElement(source, 'database')
        database.text = self.databaseSrc

        size_part = SubElement(top, 'size')
        width = SubElement(size_part, 'width')
        height = SubElement(size_part, 'height')
        depth = SubElement(size_part, 'depth')
        width.text = str(self.imgSize[1])
        height.text = str(self.imgSize[0])
        if len(self.imgSize) == 3:
            depth.text = str(self.imgSize[2])
        else:
            depth.text = '1'

        segmented = SubElement(top, 'segmented')
        segmented.text = '0'
        return top

    def addBndBox(self, xmin, ymin, xmax, ymax, 
        # name, gender, emotion, blurriness, frontal, yaw, roll, pitch):
        name, gender, blurriness, frontal, yaw, roll, pitch):
        bndbox = {'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax}
        bndbox['name'] = name
        bndbox['gender'] = gender
        # bndbox['emotion'] = emotion
        bndbox['blurriness'] = blurriness
        bndbox['frontal'] = frontal
        bndbox['yaw'] = yaw
        bndbox['roll'] = roll
        bndbox['pitch'] = pitch
        self.boxlist.append(bndbox)

    def appendObjects(self, top):
        for each_object in self.boxlist:
            object_item = SubElement(top, 'object')
            name = SubElement(object_item, 'name')
            try:
                name.text = unicode(each_object['name'])
            except NameError:
                # Py3: NameError: name 'unicode' is not defined
                name.text = each_object['name']

            truncated = SubElement(object_item, 'truncated')
            if int(each_object['ymax']) == int(self.imgSize[0]) or (int(each_object['ymin'])== 1):
                truncated.text = "1" # max == height or min
            elif (int(each_object['xmax'])==int(self.imgSize[1])) or (int(each_object['xmin'])== 1):
                truncated.text = "1" # max == width or min
            else:
                truncated.text = "0"

            gender = SubElement(object_item, 'gender')
            gender.text = str(bool(each_object['gender']) & 1 )

            blurriness = SubElement(object_item, 'blurriness')
            blurriness.text = str(bool(each_object['blurriness']) & 1 )

            frontal = SubElement(object_item, 'frontal')
            frontal.text = str(each_object['frontal'])
            
            yaw = SubElement(object_item, 'yaw')
            yaw.text = str(each_object['yaw'])

            roll = SubElement(object_item, 'roll')
            roll.text = str(each_object['roll'])

            pitch = SubElement(object_item, 'pitch')
            pitch.text = str(each_object['pitch'])

            bndbox = SubElement(object_item, 'bndbox')
            xmin = SubElement(bndbox, 'xmin')
            xmin.text = str(each_object['xmin'])
            ymin = SubElement(bndbox, 'ymin')
            ymin.text = str(each_object['ymin'])
            xmax = SubElement(bndbox, 'xmax')
            xmax.text = str(each_object['xmax'])
            ymax = SubElement(bndbox, 'ymax')
            ymax.text = str(each_object['ymax'])

    def save(self, targetFile=None):
        root = self.genXML()
        self.appendObjects(root)
        out_file = None
        if targetFile is None:
            out_file = codecs.open(
                self.filename + XML_EXT, 'w', encoding=ENCODE_METHOD)
        else:
            out_file = codecs.open(targetFile, 'w', encoding=ENCODE_METHOD)

        prettifyResult = self.prettify(root)
        out_file.write(prettifyResult.decode('utf8'))
        out_file.close()


class PascalVocReader:

    def __init__(self, filepath):
        # shapes type:
        # [labbel, [(x1,y1), (x2,y2), (x3,y3), (x4,y4)], color, color, ismale]
        self.shapes = []
        self.filepath = filepath
        self.verified = False
        try:
            self.parseXML()
        except:
            pass

    def getShapes(self):
        return self.shapes

    def addShape(self, label, bndbox,
                 ismale, isfemale, noblur, blur,
                 face_visible, face_occlusion, face_truncated, face_invisible,
                 yaw, roll, pitch):
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        points = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
        self.shapes.append((label, points, None, None,
                            ismale, isfemale, noblur, blur,
                            face_visible, face_occlusion, face_truncated, face_invisible,
                            yaw, roll, pitch
                            ))

    def parseXML(self):
        assert self.filepath.endswith(XML_EXT), "Unsupport file format"
        parser = etree.XMLParser(encoding=ENCODE_METHOD)
        xmltree = ElementTree.parse(self.filepath, parser=parser).getroot()
        if xmltree.find('filename') is not None:
            filename = xmltree.find('filename').text
        try:
            verified = xmltree.attrib['verified']
            if verified == 'yes':
                self.verified = True
        except KeyError:
            self.verified = False

        for object_iter in xmltree.findall('object'):
            bndbox = object_iter.find("bndbox")
            label = object_iter.find('name').text

            ismale = False
            isfemale = False
            
            noblur = False
            blur = False
            
            face_visible = False
            face_occlusion = False
            face_truncated = False
            face_invisible = False
            
            yaw = 0
            roll = 0
            pitch = 0
            

            if object_iter.find('gender') is not None:
                ismale = bool(int(object_iter.find('gender').text))
                isfemale = not ismale

            if object_iter.find('blurriness') is not None:
                blur = bool(int(object_iter.find('blurriness').text))
                noblur = not blur

            if object_iter.find('frontal') is not None:
                if int(object_iter.find('frontal').text) == 1:
                    face_occlusion = True
                elif int(object_iter.find('frontal').text) == 2:
                    face_truncated = True
                elif int(object_iter.find('frontal').text) == 3:
                    face_invisible = True
                else:
                    face_visible = True

            if object_iter.find('yaw') is not None:
                yaw = round(float(object_iter.find('yaw').text))
            if object_iter.find('roll') is not None:
                roll = round(float(object_iter.find('roll').text))
            if object_iter.find('pitch') is not None:
                pitch = round(float(object_iter.find('pitch').text))
                
            self.addShape(label, bndbox,
                          ismale, isfemale, noblur, blur,
                          face_visible, face_occlusion, face_truncated, face_invisible,
                          yaw, roll, pitch)
        return True
