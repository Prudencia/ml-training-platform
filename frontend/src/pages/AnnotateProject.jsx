import React, { useState, useEffect, useCallback, useRef } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { Stage, Layer, Rect, Image as KonvaImage, Transformer, Line } from 'react-konva'
import useImage from 'use-image'
import {
  ArrowLeft, ArrowRight, Plus, Trash2, X, Upload, ChevronLeft,
  Settings, Shuffle, Download, Check, ZoomIn, ZoomOut, FolderOpen, Grid, Eye, Loader2,
  Edit3, MoreVertical, AlertTriangle, CheckCircle, AlertCircle
} from 'lucide-react'
import { useDropzone } from 'react-dropzone'
import { annotationsAPI } from '../services/api'

// Class colors
const CLASS_COLORS = [
  '#EF4444', '#F59E0B', '#10B981', '#3B82F6', '#8B5CF6',
  '#EC4899', '#06B6D4', '#84CC16', '#F97316', '#6366F1'
]

function AnnotateProject() {
  const { projectId } = useParams()
  const navigate = useNavigate()

  // Project state
  const [project, setProject] = useState(null)
  const [classes, setClasses] = useState([])
  const [images, setImages] = useState([])
  const [totalImages, setTotalImages] = useState(0)
  const [currentIndex, setCurrentIndex] = useState(0)
  const [loading, setLoading] = useState(true)
  const [loadingMore, setLoadingMore] = useState(false)
  const [currentPage, setCurrentPage] = useState(1)
  const [pageStartIndex, setPageStartIndex] = useState(0) // Absolute index of first image in current page
  const PAGE_SIZE = 500

  // Annotation state
  const [annotations, setAnnotations] = useState([])
  const [selectedClassIdx, setSelectedClassIdx] = useState(0)  // Array index, not class_index
  const [selectedAnnotations, setSelectedAnnotations] = useState(new Set())
  const [autoAdvance, setAutoAdvance] = useState(true)
  const [classFilter, setClassFilter] = useState('all')  // all, unannotated, annotated, has:X, missing:X

  // Drawing state
  const [isDrawing, setIsDrawing] = useState(false)
  const [newRect, setNewRect] = useState(null)

  // Canvas state
  const [zoom, setZoom] = useState(1)
  const [canvasSize, setCanvasSize] = useState({ width: 800, height: 600 })
  const [imageScale, setImageScale] = useState({ scale: 1, offsetX: 0, offsetY: 0 })
  const [showGrid, setShowGrid] = useState(false)

  // Modals
  const [showUploadModal, setShowUploadModal] = useState(false)
  const [showSettingsModal, setShowSettingsModal] = useState(false)
  const [showSplitModal, setShowSplitModal] = useState(false)
  const [showAddClassModal, setShowAddClassModal] = useState(false)

  // Upload state
  const [uploading, setUploading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)

  // Split config
  const [splitConfig, setSplitConfig] = useState({
    train_ratio: 0.7,
    val_ratio: 0.2,
    test_ratio: 0.1,
    method: 'random',
    seed: 42
  })

  // Preprocessing config
  const [preprocessConfig, setPreprocessConfig] = useState({
    resize_enabled: false,
    target_size: 640,
    letterbox: true,
    letterbox_color: '#000000',
    auto_orient: true,
    normalize: false,
    clahe_enabled: false,
    clahe_clip_limit: 2.0
  })

  // Augmentation config (applied at export)
  const [augmentConfig, setAugmentConfig] = useState({
    enabled: false,
    copies: 2,
    horizontal_flip: true,
    vertical_flip: false,
    rotation_range: 15,
    brightness_range: 0.2,
    contrast_range: 0.2,
    saturation_range: 0.2,
    noise_enabled: false,
    blur_enabled: false
  })

  // New class state
  const [newClassName, setNewClassName] = useState('')
  const [newClassColor, setNewClassColor] = useState('#EF4444')
  const [showManageClassesModal, setShowManageClassesModal] = useState(false)
  const [editingClass, setEditingClass] = useState(null)
  const [editClassName, setEditClassName] = useState('')
  const [editClassColor, setEditClassColor] = useState('')
  const [showRemapModal, setShowRemapModal] = useState(false)
  const [annotationClassStats, setAnnotationClassStats] = useState(null)
  const [remapMappings, setRemapMappings] = useState({})
  const [remapLoading, setRemapLoading] = useState(false)

  // Augmentation preview state
  const [augmentPreview, setAugmentPreview] = useState(null)
  const [augmentPreviewLoading, setAugmentPreviewLoading] = useState(false)

  // Quality check state
  const [showQualityModal, setShowQualityModal] = useState(false)
  const [qualityResults, setQualityResults] = useState(null)
  const [qualityLoading, setQualityLoading] = useState(false)
  const [fixLoading, setFixLoading] = useState(null) // Track which fix is running
  const [fixResult, setFixResult] = useState(null) // Show fix results

  // Canvas container ref
  const containerRef = useRef(null)
  const transformerRef = useRef(null)
  const stageRef = useRef(null)

  // Current image - use scaled version (1920px max) for faster loading
  const VIEW_SIZE = 1920  // Max dimension for annotation viewing
  const currentImage = images[currentIndex]
  const absoluteIndex = pageStartIndex + currentIndex // Absolute position in the full dataset
  const [image, imageStatus] = useImage(
    currentImage ? annotationsAPI.getImageUrl(projectId, currentImage.id, VIEW_SIZE) : null,
    'anonymous'
  )

  // Preload next and previous images for faster navigation
  const nextImage = images[currentIndex + 1]
  const prevImage = images[currentIndex - 1]
  useImage(nextImage ? annotationsAPI.getImageUrl(projectId, nextImage.id, VIEW_SIZE) : null, 'anonymous')
  useImage(prevImage ? annotationsAPI.getImageUrl(projectId, prevImage.id, VIEW_SIZE) : null, 'anonymous')
  // Preload 2 ahead for even smoother navigation
  const nextImage2 = images[currentIndex + 2]
  useImage(nextImage2 ? annotationsAPI.getImageUrl(projectId, nextImage2.id, VIEW_SIZE) : null, 'anonymous')

  // Load project data
  useEffect(() => {
    loadProject()
  }, [projectId])

  // Update canvas size on resize
  useEffect(() => {
    const updateSize = () => {
      if (containerRef.current) {
        const rect = containerRef.current.getBoundingClientRect()
        setCanvasSize({ width: rect.width, height: rect.height })
      }
    }
    updateSize()
    window.addEventListener('resize', updateSize)
    return () => window.removeEventListener('resize', updateSize)
  }, [])

  // Calculate image scale for letterboxing
  useEffect(() => {
    if (image && currentImage) {
      const imgWidth = currentImage.original_width
      const imgHeight = currentImage.original_height
      const containerW = canvasSize.width
      const containerH = canvasSize.height

      const scale = Math.min(containerW / imgWidth, containerH / imgHeight) * zoom
      const offsetX = (containerW - imgWidth * scale) / 2
      const offsetY = (containerH - imgHeight * scale) / 2

      setImageScale({ scale, offsetX, offsetY })
    }
  }, [image, currentImage, canvasSize, zoom])

  // Load annotations when image changes
  useEffect(() => {
    if (currentImage) {
      loadAnnotations(currentImage.id)
    }
  }, [currentImage])

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return

      switch (e.key) {
        case 'ArrowRight':
        case 'd':
          e.preventDefault()
          goToNext()
          break
        case 'ArrowLeft':
        case 'a':
          e.preventDefault()
          goToPrevious()
          break
        case 'Delete':
        case 'Backspace':
          e.preventDefault()
          if (e.shiftKey && images[currentIndex]) {
            // Shift+Delete = delete image
            const img = images[currentIndex]
            if (confirm(`Delete image "${img.filename}"?`)) {
              annotationsAPI.deleteImage(projectId, img.id).then(() => {
                const newImages = images.filter(i => i.id !== img.id)
                setImages(newImages)
                if (newImages.length === 0) {
                  setCurrentIndex(0)
                  setAnnotations([])
                } else if (currentIndex >= newImages.length) {
                  setCurrentIndex(newImages.length - 1)
                }
              }).catch(err => {
                alert('Failed to delete: ' + (err.response?.data?.detail || err.message))
              })
            }
          } else if (selectedAnnotations.size > 0) {
            // Delete all selected annotations
            deleteSelectedAnnotations()
          }
          break
        case 'Tab':
          e.preventDefault()
          setAutoAdvance(!autoAdvance)
          break
        case 'Escape':
          e.preventDefault()
          // Clear selection
          setSelectedAnnotations(new Set())
          break
        case '1': case '2': case '3': case '4': case '5':
        case '6': case '7': case '8': case '9':
          const keyIdx = parseInt(e.key) - 1
          if (keyIdx < classes.length) {
            if (selectedAnnotations.size > 0) {
              // Change class of selected annotations - use class_index from the target class
              changeSelectedAnnotationsClass(classes[keyIdx].class_index)
            } else {
              // Just select the class for drawing - use array index
              setSelectedClassIdx(keyIdx)
            }
          }
          break
        case '=':
        case '+':
          setZoom(z => Math.min(z + 0.1, 3))
          break
        case '-':
          setZoom(z => Math.max(z - 0.1, 0.3))
          break
        case '0':
          setZoom(1)
          break
        case 'g':
        case 'G':
          setShowGrid(g => !g)
          break
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [classes, selectedAnnotations, autoAdvance, currentIndex, images, projectId, annotations])

  // Helper to parse filter value into API parameters
  const parseFilterParams = (filter) => {
    let filterStatus = null
    let hasClass = null
    let missingClass = null

    if (filter === 'unannotated') {
      filterStatus = 'unannotated'
    } else if (filter === 'annotated') {
      filterStatus = 'annotated'
    } else if (filter.startsWith('has:')) {
      hasClass = parseInt(filter.split(':')[1])
    } else if (filter.startsWith('missing:')) {
      missingClass = parseInt(filter.split(':')[1])
    }

    return { filterStatus, hasClass, missingClass }
  }

  // Helper to load images with current filter
  const loadImagesWithFilter = async (page = 1, filter = classFilter) => {
    const { filterStatus, hasClass, missingClass } = parseFilterParams(filter)
    return annotationsAPI.listImages(projectId, page, PAGE_SIZE, filterStatus, hasClass, missingClass)
  }

  const loadProject = async () => {
    try {
      setLoading(true)
      const [projectRes, imagesRes] = await Promise.all([
        annotationsAPI.getProject(projectId),
        loadImagesWithFilter(1, classFilter)
      ])

      setProject(projectRes.data)
      const projectClasses = Array.isArray(projectRes.data?.classes) ? projectRes.data.classes : []
      const projectImages = Array.isArray(imagesRes.data?.images) ? imagesRes.data.images : []
      // Sort classes by class_index to ensure keyboard shortcuts match display order
      setClasses([...projectClasses].sort((a, b) => a.class_index - b.class_index))
      setImages(projectImages)
      setTotalImages(imagesRes.data?.total || projectImages.length)
      setCurrentPage(1)

      // Load preprocessing config from project
      if (projectRes.data?.preprocessing_config) {
        setPreprocessConfig(prev => ({ ...prev, ...projectRes.data.preprocessing_config }))
      }
      if (projectRes.data?.augmentation_config) {
        setAugmentConfig(prev => ({ ...prev, ...projectRes.data.augmentation_config }))
      }

      // Load annotation class stats (to find orphaned classes)
      try {
        const statsRes = await annotationsAPI.getAnnotationClassStats(projectId)
        setAnnotationClassStats(statsRes.data)
      } catch (err) {
        console.warn('Failed to load annotation class stats:', err)
      }

      // Resume from last position
      const resumeRes = await annotationsAPI.getNextUnannotated(projectId)
      if (resumeRes.data?.image_id && projectImages.length > 0) {
        const idx = projectImages.findIndex(img => img.id === resumeRes.data.image_id)
        if (idx >= 0) setCurrentIndex(idx)
      }
    } catch (error) {
      console.error('Failed to load project:', error)
      alert('Failed to load project')
      navigate('/annotate')
    } finally {
      setLoading(false)
    }
  }

  // Load more images (pagination) - only appends when starting from beginning
  const loadMoreImages = async () => {
    // Only auto-load if we started from page 1 (sequential browsing)
    if (loadingMore || pageStartIndex !== 0 || pageStartIndex + images.length >= totalImages) return

    setLoadingMore(true)
    try {
      const nextPage = currentPage + 1
      const res = await loadImagesWithFilter(nextPage)
      const newImages = Array.isArray(res.data?.images) ? res.data.images : []

      if (newImages.length > 0) {
        setImages(prev => [...prev, ...newImages])
        setCurrentPage(nextPage)
      }
    } catch (error) {
      console.error('Failed to load more images:', error)
    } finally {
      setLoadingMore(false)
    }
  }

  // Auto-load more images when approaching end (only if starting from beginning)
  useEffect(() => {
    if (pageStartIndex === 0 && currentIndex >= images.length - 50 && pageStartIndex + images.length < totalImages && !loadingMore) {
      loadMoreImages()
    }
  }, [currentIndex, images.length, totalImages, loadingMore, pageStartIndex])

  const loadAnnotations = async (imageId) => {
    try {
      const res = await annotationsAPI.getAnnotations(imageId)
      setAnnotations(Array.isArray(res.data?.annotations) ? res.data.annotations : [])
      setSelectedAnnotations(new Set())
    } catch (error) {
      console.error('Failed to load annotations:', error)
      setAnnotations([])
    }
  }

  const saveAnnotations = useCallback(async (newAnnotations) => {
    if (!currentImage) return
    try {
      await annotationsAPI.saveAnnotations(currentImage.id, newAnnotations)
      // Update local image status
      const updatedImages = [...images]
      updatedImages[currentIndex] = {
        ...updatedImages[currentIndex],
        is_annotated: newAnnotations.length > 0,
        annotation_count: newAnnotations.length
      }
      setImages(updatedImages)
    } catch (error) {
      console.error('Failed to save annotations:', error)
    }
  }, [currentImage, currentIndex, images])

  const goToNext = useCallback(() => {
    const nextAbsoluteIndex = pageStartIndex + currentIndex + 1
    if (nextAbsoluteIndex < totalImages) {
      if (currentIndex < images.length - 1) {
        // Still in current page
        setCurrentIndex(currentIndex + 1)
      } else {
        // Need to load next page
        handleGoToImage(nextAbsoluteIndex)
      }
    }
  }, [currentIndex, images.length, pageStartIndex, totalImages])

  const goToPrevious = useCallback(() => {
    const prevAbsoluteIndex = pageStartIndex + currentIndex - 1
    if (prevAbsoluteIndex >= 0) {
      if (currentIndex > 0) {
        // Still in current page
        setCurrentIndex(currentIndex - 1)
      } else {
        // Need to load previous page
        handleGoToImage(prevAbsoluteIndex)
      }
    }
  }, [currentIndex, pageStartIndex])

  // Convert canvas coordinates to normalized YOLO format
  const canvasToYolo = (rect) => {
    const { scale, offsetX, offsetY } = imageScale
    const imgW = currentImage.original_width
    const imgH = currentImage.original_height

    const x = (rect.x - offsetX) / scale
    const y = (rect.y - offsetY) / scale
    const w = rect.width / scale
    const h = rect.height / scale

    return {
      x_center: (x + w / 2) / imgW,
      y_center: (y + h / 2) / imgH,
      width: w / imgW,
      height: h / imgH
    }
  }

  // Convert YOLO format to canvas coordinates
  const yoloToCanvas = (ann) => {
    const { scale, offsetX, offsetY } = imageScale
    const imgW = currentImage.original_width
    const imgH = currentImage.original_height

    const w = ann.width * imgW * scale
    const h = ann.height * imgH * scale
    const x = (ann.x_center * imgW - ann.width * imgW / 2) * scale + offsetX
    const y = (ann.y_center * imgH - ann.height * imgH / 2) * scale + offsetY

    return { x, y, width: w, height: h }
  }

  // Mouse handlers for drawing
  const handleMouseDown = (e) => {
    if (classes.length === 0) {
      alert('Please add at least one class first')
      return
    }

    const stage = e.target.getStage()
    const pos = stage.getPointerPosition()

    // Check if clicking on existing annotation
    const clickedOnEmpty = e.target === e.target.getStage() ||
      e.target.className === 'Image'

    if (clickedOnEmpty) {
      setSelectedAnnotations(new Set())
      setIsDrawing(true)
      setNewRect({
        x: pos.x,
        y: pos.y,
        width: 0,
        height: 0
      })
    }
  }

  const handleMouseMove = (e) => {
    if (!isDrawing || !newRect) return

    const stage = e.target.getStage()
    const pos = stage.getPointerPosition()

    setNewRect({
      ...newRect,
      width: pos.x - newRect.x,
      height: pos.y - newRect.y
    })
  }

  const handleMouseUp = () => {
    if (!isDrawing || !newRect) return

    setIsDrawing(false)

    // Normalize rect (handle negative dimensions)
    let { x, y, width, height } = newRect
    if (width < 0) {
      x = x + width
      width = -width
    }
    if (height < 0) {
      y = y + height
      height = -height
    }

    // Minimum size check
    if (width < 10 || height < 10) {
      setNewRect(null)
      return
    }

    // Convert to YOLO format
    const yolo = canvasToYolo({ x, y, width, height })

    // Validate bounds
    if (yolo.x_center < 0 || yolo.x_center > 1 ||
        yolo.y_center < 0 || yolo.y_center > 1 ||
        yolo.width <= 0 || yolo.height <= 0) {
      setNewRect(null)
      return
    }

    // Add annotation - use class_index from the selected class
    const newAnnotation = {
      class_id: classes[selectedClassIdx]?.class_index ?? 0,
      ...yolo
    }

    const newAnnotations = [...annotations, newAnnotation]
    setAnnotations(newAnnotations)
    saveAnnotations(newAnnotations)
    setNewRect(null)

    // Auto-advance
    if (autoAdvance && currentIndex < images.length - 1) {
      setTimeout(() => goToNext(), 200)
    }
  }

  const deleteAnnotation = (index) => {
    const newAnnotations = annotations.filter((_, i) => i !== index)
    setAnnotations(newAnnotations)
    saveAnnotations(newAnnotations)
    // Update selected annotations - remove the deleted one and adjust indices
    setSelectedAnnotations(prev => {
      const updated = new Set()
      prev.forEach(idx => {
        if (idx < index) updated.add(idx)
        else if (idx > index) updated.add(idx - 1)
        // idx === index is deleted, so skip
      })
      return updated
    })
  }

  // Delete multiple selected annotations
  const deleteSelectedAnnotations = () => {
    if (selectedAnnotations.size === 0) return
    const indicesToDelete = Array.from(selectedAnnotations).sort((a, b) => b - a) // Sort descending to delete from end
    let newAnnotations = [...annotations]
    indicesToDelete.forEach(idx => {
      newAnnotations.splice(idx, 1)
    })
    setAnnotations(newAnnotations)
    saveAnnotations(newAnnotations)
    setSelectedAnnotations(new Set())
  }

  // Change class of selected annotations
  const changeSelectedAnnotationsClass = (newClassId) => {
    if (selectedAnnotations.size === 0) return
    const newAnnotations = annotations.map((ann, idx) => {
      if (selectedAnnotations.has(idx)) {
        return { ...ann, class_id: newClassId }
      }
      return ann
    })
    setAnnotations(newAnnotations)
    saveAnnotations(newAnnotations)
    setSelectedAnnotations(new Set())

    // Auto-advance after changing class
    if (autoAdvance && currentIndex < images.length - 1) {
      setTimeout(() => goToNext(), 200)
    }
  }

  const handleAnnotationClick = (index) => {
    // Toggle selection
    setSelectedAnnotations(prev => {
      const updated = new Set(prev)
      if (updated.has(index)) {
        updated.delete(index)
      } else {
        updated.add(index)
      }
      return updated
    })
  }

  // File upload
  const onDrop = useCallback(async (acceptedFiles) => {
    if (acceptedFiles.length === 0) return

    setUploading(true)
    setUploadProgress(0)

    const formData = new FormData()
    acceptedFiles.forEach(file => {
      formData.append('files', file)
    })

    try {
      await annotationsAPI.uploadImages(projectId, formData, (progressEvent) => {
        const percent = Math.round((progressEvent.loaded * 100) / progressEvent.total)
        setUploadProgress(percent)
      })

      // Reload images
      const imagesRes = await annotationsAPI.listImages(projectId, 1, 1000)
      setImages(Array.isArray(imagesRes.data?.images) ? imagesRes.data.images : [])

      setShowUploadModal(false)
    } catch (error) {
      alert('Failed to upload images: ' + (error.response?.data?.detail || error.message))
    } finally {
      setUploading(false)
      setUploadProgress(0)
    }
  }, [projectId])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'image/*': ['.jpg', '.jpeg', '.png', '.bmp', '.gif'] }
  })

  // Class management
  const handleAddClass = async () => {
    if (!newClassName.trim()) return
    try {
      const res = await annotationsAPI.createClass(projectId, {
        name: newClassName.trim(),
        color: newClassColor
      })
      // Sort classes after adding to maintain class_index order
      setClasses([...classes, res.data].sort((a, b) => a.class_index - b.class_index))
      setNewClassName('')
      setNewClassColor(CLASS_COLORS[(classes.length + 1) % CLASS_COLORS.length])
      setShowAddClassModal(false)
    } catch (error) {
      alert('Failed to add class: ' + (error.response?.data?.detail || error.message))
    }
  }

  const handleUpdateClass = async () => {
    if (!editingClass || !editClassName.trim()) return
    try {
      await annotationsAPI.updateClass(projectId, editingClass.id, {
        name: editClassName.trim(),
        color: editClassColor
      })
      // Reload classes and sort by class_index
      const projectRes = await annotationsAPI.getProject(projectId)
      const loadedClasses = Array.isArray(projectRes.data?.classes) ? projectRes.data.classes : []
      setClasses([...loadedClasses].sort((a, b) => a.class_index - b.class_index))
      setEditingClass(null)
      setEditClassName('')
      setEditClassColor('')
    } catch (error) {
      alert('Failed to update class: ' + (error.response?.data?.detail || error.message))
    }
  }

  const handleDeleteClass = async (classId) => {
    if (!confirm('Delete this class and all its annotations?')) return
    try {
      await annotationsAPI.deleteClass(projectId, classId, true)
      // Reload project to get updated classes and sort by class_index
      const projectRes = await annotationsAPI.getProject(projectId)
      const loadedClasses = Array.isArray(projectRes.data?.classes) ? projectRes.data.classes : []
      setClasses([...loadedClasses].sort((a, b) => a.class_index - b.class_index))
      // Reload current annotations
      if (currentImage) {
        loadAnnotations(currentImage.id)
      }
      // Refresh annotation class stats
      const statsRes = await annotationsAPI.getAnnotationClassStats(projectId)
      setAnnotationClassStats(statsRes.data)
    } catch (error) {
      alert('Failed to delete class: ' + (error.response?.data?.detail || error.message))
    }
  }

  const handleDeleteOrphanedClassAnnotations = async (classId, count) => {
    if (!confirm(`Delete all ${count} annotations with orphaned class ID ${classId}?`)) return
    try {
      await annotationsAPI.deleteAnnotationsByClass(projectId, classId)
      // Reload current annotations
      if (currentImage) {
        loadAnnotations(currentImage.id)
      }
      // Refresh annotation class stats
      const statsRes = await annotationsAPI.getAnnotationClassStats(projectId)
      setAnnotationClassStats(statsRes.data)
    } catch (error) {
      alert('Failed to delete orphaned annotations: ' + (error.response?.data?.detail || error.message))
    }
  }

  const startEditingClass = (cls) => {
    setEditingClass(cls)
    setEditClassName(cls.name)
    setEditClassColor(cls.color || CLASS_COLORS[cls.class_index % CLASS_COLORS.length])
  }

  // Bulk Remap Annotations
  const handleOpenRemapModal = async () => {
    setRemapLoading(true)
    try {
      const res = await annotationsAPI.getAnnotationClassStats(projectId)
      setAnnotationClassStats(res.data)
      // Initialize mappings to keep same class
      const initialMappings = {}
      res.data.classes?.forEach(cls => {
        initialMappings[cls.class_id] = cls.class_id
      })
      setRemapMappings(initialMappings)
      setShowRemapModal(true)
    } catch (error) {
      alert('Failed to load annotation stats: ' + (error.response?.data?.detail || error.message))
    } finally {
      setRemapLoading(false)
    }
  }

  const handleRemapMappingChange = (oldClassId, newClassId) => {
    setRemapMappings(prev => ({
      ...prev,
      [oldClassId]: parseInt(newClassId)
    }))
  }

  const handleApplyRemap = async () => {
    // Filter out mappings where old == new (no change)
    const changedMappings = {}
    Object.entries(remapMappings).forEach(([oldId, newId]) => {
      if (parseInt(oldId) !== parseInt(newId)) {
        changedMappings[oldId] = newId
      }
    })

    if (Object.keys(changedMappings).length === 0) {
      alert('No changes to apply')
      return
    }

    setRemapLoading(true)
    try {
      const res = await annotationsAPI.bulkRemapAnnotations(projectId, changedMappings)
      alert(res.data.message || `Updated ${res.data.updated_count} annotations`)
      setShowRemapModal(false)
      // Reload current annotations to reflect changes
      if (currentImage) {
        loadAnnotations(currentImage.id)
      }
    } catch (error) {
      alert('Failed to remap annotations: ' + (error.response?.data?.detail || error.message))
    } finally {
      setRemapLoading(false)
    }
  }

  // Save preprocessing and augmentation settings
  const handleSaveSettings = async () => {
    try {
      await annotationsAPI.updateProject(projectId, {
        preprocessing_config: preprocessConfig,
        augmentation_config: augmentConfig
      })
      alert('Settings saved successfully')
      setShowSettingsModal(false)
    } catch (error) {
      alert('Failed to save settings: ' + (error.response?.data?.detail || error.message))
    }
  }

  // Preview augmentation
  const handleAugmentationPreview = async () => {
    if (!currentImage || !augmentConfig.enabled) return

    setAugmentPreviewLoading(true)
    try {
      const res = await annotationsAPI.getAugmentationPreview(projectId, currentImage.id, augmentConfig)
      setAugmentPreview(res.data)
    } catch (error) {
      console.error('Failed to get augmentation preview:', error)
      alert('Failed to generate preview: ' + (error.response?.data?.detail || error.message))
    } finally {
      setAugmentPreviewLoading(false)
    }
  }

  // Quality check
  const handleQualityCheck = async () => {
    setQualityLoading(true)
    setShowQualityModal(true)
    setFixResult(null)
    try {
      const res = await annotationsAPI.qualityCheck(projectId)
      setQualityResults(res.data)
    } catch (error) {
      console.error('Failed to run quality check:', error)
      alert('Failed to run quality check: ' + (error.response?.data?.detail || error.message))
      setShowQualityModal(false)
    } finally {
      setQualityLoading(false)
    }
  }

  // Quick Fix handlers
  const handleFixSmallBoxes = async () => {
    if (!confirm('Delete all boxes smaller than 1% of image area? This cannot be undone.')) return
    setFixLoading('small')
    try {
      const res = await annotationsAPI.fixDeleteSmallBoxes(projectId)
      setFixResult({ type: 'success', message: `Deleted ${res.data.deleted} small boxes from ${res.data.affected_images} images` })
      handleQualityCheck() // Refresh results
    } catch (error) {
      setFixResult({ type: 'error', message: error.response?.data?.detail || error.message })
    } finally {
      setFixLoading(null)
    }
  }

  const handleFixLargeBoxes = async () => {
    if (!confirm('Delete all boxes larger than 90% of image area? This cannot be undone.')) return
    setFixLoading('large')
    try {
      const res = await annotationsAPI.fixDeleteLargeBoxes(projectId)
      setFixResult({ type: 'success', message: `Deleted ${res.data.deleted} large boxes from ${res.data.affected_images} images` })
      handleQualityCheck() // Refresh results
    } catch (error) {
      setFixResult({ type: 'error', message: error.response?.data?.detail || error.message })
    } finally {
      setFixLoading(null)
    }
  }

  const handleFixOverlapping = async () => {
    if (!confirm('Merge overlapping boxes (IoU > 80%) by keeping the larger one? This cannot be undone.')) return
    setFixLoading('overlap')
    try {
      const res = await annotationsAPI.fixMergeOverlapping(projectId)
      setFixResult({ type: 'success', message: `Removed ${res.data.deleted} duplicate boxes from ${res.data.affected_images} images` })
      handleQualityCheck() // Refresh results
    } catch (error) {
      setFixResult({ type: 'error', message: error.response?.data?.detail || error.message })
    } finally {
      setFixLoading(null)
    }
  }

  const handleGoToUnannotated = async () => {
    try {
      const res = await annotationsAPI.getUnannotatedImages(projectId)
      if (res.data.images.length > 0) {
        const firstUnannotated = res.data.images[0]
        // Find index in current images
        const idx = images.findIndex(img => img.id === firstUnannotated.id)
        if (idx >= 0) {
          setCurrentIndex(idx)
          setShowQualityModal(false)
        } else {
          alert(`First unannotated image: ${firstUnannotated.filename} (ID: ${firstUnannotated.id}).\nYou may need to navigate to a different page.`)
        }
      } else {
        alert('No unannotated images found!')
      }
    } catch (error) {
      alert('Failed to find unannotated images: ' + (error.response?.data?.detail || error.message))
    }
  }

  // Jump to specific image (absolute index)
  const handleGoToImage = (absoluteIndex) => {
    if (absoluteIndex >= 0 && absoluteIndex < totalImages) {
      // Calculate which page this index is on
      const targetPage = Math.floor(absoluteIndex / PAGE_SIZE) + 1
      const targetPageStart = (targetPage - 1) * PAGE_SIZE
      const indexInPage = absoluteIndex - targetPageStart

      // Check if we already have this page loaded
      if (absoluteIndex >= pageStartIndex && absoluteIndex < pageStartIndex + images.length) {
        // Already loaded, just change index
        setCurrentIndex(absoluteIndex - pageStartIndex)
      } else {
        // Need to load the page containing this index
        setLoadingMore(true)
        loadImagesWithFilter(targetPage).then(res => {
          const newImages = Array.isArray(res.data?.images) ? res.data.images : []
          setImages(newImages)
          setCurrentPage(targetPage)
          setPageStartIndex(targetPageStart)
          setCurrentIndex(indexInPage)
          setLoadingMore(false)
        }).catch(err => {
          console.error('Failed to jump to image:', err)
          setLoadingMore(false)
        })
      }
    }
  }

  // Handle filter change - reload images from page 1
  const handleFilterChange = async (newFilter) => {
    setClassFilter(newFilter)
    setLoading(true)
    try {
      const res = await loadImagesWithFilter(1, newFilter)
      const newImages = Array.isArray(res.data?.images) ? res.data.images : []
      setImages(newImages)
      setTotalImages(res.data?.total || newImages.length)
      setCurrentPage(1)
      setPageStartIndex(0)
      setCurrentIndex(0)
    } catch (error) {
      console.error('Failed to apply filter:', error)
    } finally {
      setLoading(false)
    }
  }

  // Split generation
  const handleGenerateSplits = async () => {
    try {
      const res = await annotationsAPI.generateSplits(projectId, splitConfig)
      alert(`Splits generated: Train ${res.data.train_count}, Val ${res.data.val_count}, Test ${res.data.test_count}`)
      setShowSplitModal(false)
    } catch (error) {
      alert('Failed to generate splits: ' + (error.response?.data?.detail || error.message))
    }
  }

  // Delete current image
  const handleDeleteImage = async () => {
    if (!currentImage) return
    if (!confirm(`Delete this image "${currentImage.filename}"? This will also remove any annotations.`)) return

    try {
      await annotationsAPI.deleteImage(projectId, currentImage.id)

      // Remove from local state and adjust index
      const newImages = images.filter(img => img.id !== currentImage.id)
      setImages(newImages)

      // Adjust current index
      if (newImages.length === 0) {
        setCurrentIndex(0)
        setAnnotations([])
      } else if (currentIndex >= newImages.length) {
        setCurrentIndex(newImages.length - 1)
      }
    } catch (error) {
      alert('Failed to delete image: ' + (error.response?.data?.detail || error.message))
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="text-gray-500">Loading project...</div>
      </div>
    )
  }

  return (
    <div className="h-[calc(100vh-8rem)] flex flex-col">
      {/* Header */}
      <div className="bg-white shadow px-4 py-2 flex items-center justify-between">
        <div className="flex items-center gap-4">
          <button
            onClick={() => navigate('/annotate')}
            className="text-gray-600 hover:text-gray-900"
          >
            <ChevronLeft size={24} />
          </button>
          <h1 className="text-lg font-bold text-emerald-600">{project?.name}</h1>
          <span className="text-sm text-gray-500">
            Image {(absoluteIndex + 1).toLocaleString()} of {totalImages.toLocaleString()}
            {loadingMore && ' (loading...)'}
          </span>
          {/* Jump to image input */}
          <input
            type="number"
            min={1}
            max={totalImages}
            placeholder="Go to #"
            className="w-24 px-2 py-1 text-sm border rounded"
            onKeyDown={(e) => {
              if (e.key === 'Enter') {
                const val = parseInt(e.target.value)
                if (val > 0 && val <= totalImages) {
                  handleGoToImage(val - 1)
                  e.target.value = ''
                }
              }
            }}
          />
        </div>

        <div className="flex items-center gap-2">
          {/* Class filter dropdown */}
          <select
            value={classFilter}
            onChange={(e) => handleFilterChange(e.target.value)}
            className="text-sm border border-gray-300 rounded px-2 py-1.5 bg-white"
          >
            <option value="all">All images</option>
            <option value="unannotated">Unannotated</option>
            <option value="annotated">Annotated</option>
            {classes.length > 0 && (
              <>
                <optgroup label="Has class">
                  {classes.map((cls, idx) => (
                    <option key={`has-${cls.class_index}`} value={`has:${cls.class_index}`}>
                      Has: {cls.name}
                    </option>
                  ))}
                </optgroup>
                <optgroup label="Missing class">
                  {classes.map((cls, idx) => (
                    <option key={`missing-${cls.class_index}`} value={`missing:${cls.class_index}`}>
                      Missing: {cls.name}
                    </option>
                  ))}
                </optgroup>
              </>
            )}
            {/* Orphaned classes (class IDs that exist in annotations but have no class definition) */}
            {annotationClassStats?.classes?.filter(c => c.is_orphaned).length > 0 && (
              <optgroup label="Orphaned (Has)">
                {annotationClassStats.classes.filter(c => c.is_orphaned).map((cls) => (
                  <option key={`has-orphan-${cls.class_id}`} value={`has:${cls.class_id}`}>
                    Has: Unknown (ID: {cls.class_id}) [{cls.count}]
                  </option>
                ))}
              </optgroup>
            )}
          </select>

          <label className="flex items-center gap-2 text-sm">
            <input
              type="checkbox"
              checked={autoAdvance}
              onChange={(e) => setAutoAdvance(e.target.checked)}
              className="rounded"
            />
            Auto-advance
          </label>

          <button
            onClick={handleQualityCheck}
            className="px-3 py-1 bg-yellow-100 text-yellow-700 rounded hover:bg-yellow-200 flex items-center gap-1"
            title="Check annotation quality"
          >
            <AlertTriangle size={16} />
            Quality
          </button>

          <button
            onClick={() => setShowSettingsModal(true)}
            className="px-3 py-1 bg-gray-100 text-gray-700 rounded hover:bg-gray-200 flex items-center gap-1"
          >
            <Settings size={16} />
            Settings
          </button>

          <button
            onClick={() => setShowUploadModal(true)}
            className="px-3 py-1 bg-blue-100 text-blue-700 rounded hover:bg-blue-200 flex items-center gap-1"
          >
            <Upload size={16} />
            Upload
          </button>

          <button
            onClick={() => setShowSplitModal(true)}
            className="px-3 py-1 bg-purple-100 text-purple-700 rounded hover:bg-purple-200 flex items-center gap-1"
          >
            <Shuffle size={16} />
            Split
          </button>
        </div>
      </div>

      {/* Main content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left sidebar - Classes */}
        <div className="w-48 bg-gray-50 border-r flex flex-col">
          <div className="p-3 border-b">
            <div className="flex items-center justify-between mb-2">
              <span className="font-medium text-sm">Classes</span>
              <button
                onClick={() => setShowAddClassModal(true)}
                className="text-emerald-600 hover:text-emerald-700 p-1"
                title="Add class"
              >
                <Plus size={16} />
              </button>
            </div>
            <button
              onClick={() => setShowManageClassesModal(true)}
              className="w-full text-xs px-2 py-1 bg-gray-200 hover:bg-gray-300 rounded flex items-center justify-center gap-1 text-gray-700"
            >
              <Settings size={12} />
              Manage Classes
            </button>
            {/* Show currently selected drawing class */}
            {classes.length > 0 && selectedClassIdx < classes.length && (() => {
              const activeClass = classes[selectedClassIdx]
              const activeColor = activeClass?.color || CLASS_COLORS[selectedClassIdx % CLASS_COLORS.length]
              return activeClass ? (
                <div
                  className="text-xs px-2 py-1.5 rounded flex items-center gap-2"
                  style={{ backgroundColor: `${activeColor}20`, color: activeColor }}
                >
                  <div className="w-3 h-3 rounded" style={{ backgroundColor: activeColor }} />
                  <span className="font-medium truncate">Drawing: {activeClass.name}</span>
                  <span className="ml-auto opacity-70">[{selectedClassIdx + 1}]</span>
                </div>
              ) : null
            })()}
          </div>

          <div className="flex-1 overflow-y-auto p-2 space-y-1">
            {classes.length === 0 ? (
              <p className="text-sm text-gray-500 text-center py-4">No classes yet</p>
            ) : (
              classes.map((cls, idx) => {
                const isSelected = selectedClassIdx === idx
                const classColor = cls.color || CLASS_COLORS[idx % CLASS_COLORS.length]
                return (
                  <button
                    key={cls.id}
                    onClick={() => setSelectedClassIdx(idx)}
                    className={`w-full flex items-center gap-2 px-3 py-2 rounded text-sm transition-all ${
                      isSelected
                        ? 'font-medium border-2'
                        : 'hover:bg-gray-100 border-2 border-transparent'
                    }`}
                    style={isSelected ? {
                      backgroundColor: `${classColor}20`,
                      borderColor: classColor,
                      color: classColor
                    } : {}}
                  >
                    <div
                      className="w-4 h-4 rounded"
                      style={{ backgroundColor: classColor }}
                    />
                    <span className="flex-1 text-left truncate">{cls.name}</span>
                    <span className={`text-xs px-1.5 py-0.5 rounded ${isSelected ? 'bg-white/50 font-bold' : 'text-gray-400'}`}>
                      {idx + 1}
                    </span>
                  </button>
                )
              })
            )}
          </div>

          {/* Shortcuts panel */}
          <div className="p-3 border-t">
            <div className="text-xs font-medium text-gray-500 mb-2">Shortcuts</div>
            <div className="space-y-1 text-xs text-gray-600">
              <div className="flex justify-between">
                <span>Navigate</span>
                <span className="font-mono bg-gray-200 px-1 rounded">← → A D</span>
              </div>
              <div className="flex justify-between">
                <span>Class / Bulk change</span>
                <span className="font-mono bg-gray-200 px-1 rounded">1-9</span>
              </div>
              <div className="flex justify-between">
                <span>Delete selected</span>
                <span className="font-mono bg-gray-200 px-1 rounded">Del</span>
              </div>
              <div className="flex justify-between">
                <span>Delete image</span>
                <span className="font-mono bg-gray-200 px-1 rounded">⇧Del</span>
              </div>
              <div className="flex justify-between">
                <span>Deselect all</span>
                <span className="font-mono bg-gray-200 px-1 rounded">Esc</span>
              </div>
              <div className="flex justify-between">
                <span>Auto-advance</span>
                <span className="font-mono bg-gray-200 px-1 rounded">Tab</span>
              </div>
              <div className="flex justify-between">
                <span>Zoom</span>
                <span className="font-mono bg-gray-200 px-1 rounded">+ - 0</span>
              </div>
              <div className="flex justify-between">
                <span>Grid</span>
                <span className="font-mono bg-gray-200 px-1 rounded">G</span>
              </div>
            </div>
          </div>

          {/* Grid & Zoom controls */}
          <div className="p-3 border-t">
            <div className="flex items-center justify-between mb-2">
              <button
                onClick={() => setShowGrid(g => !g)}
                className={`p-1.5 rounded flex items-center gap-1 text-xs ${showGrid ? 'bg-emerald-100 text-emerald-700' : 'hover:bg-gray-200 text-gray-600'}`}
                title="Toggle grid (G)"
              >
                <Grid size={14} />
                Grid
              </button>
              <div className="flex items-center gap-1">
                <button onClick={() => setZoom(z => Math.max(z - 0.1, 0.3))} className="p-1 hover:bg-gray-200 rounded">
                  <ZoomOut size={16} />
                </button>
                <span className="text-xs text-gray-600 w-10 text-center">{Math.round(zoom * 100)}%</span>
                <button onClick={() => setZoom(z => Math.min(z + 0.1, 3))} className="p-1 hover:bg-gray-200 rounded">
                  <ZoomIn size={16} />
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Canvas area */}
        <div
          ref={containerRef}
          className="flex-1 bg-gray-900 relative overflow-hidden"
        >
          {/* Loading indicator */}
          {imageStatus === 'loading' && (
            <div className="absolute inset-0 flex items-center justify-center z-10 pointer-events-none">
              <div className="bg-black/50 px-4 py-2 rounded-lg flex items-center gap-2">
                <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                <span className="text-white text-sm">Loading...</span>
              </div>
            </div>
          )}

          {images.length === 0 ? (
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="text-center text-gray-400">
                <p className="mb-4">No images in project</p>
                <button
                  onClick={() => setShowUploadModal(true)}
                  className="px-4 py-2 bg-emerald-600 text-white rounded hover:bg-emerald-700"
                >
                  Upload Images
                </button>
              </div>
            </div>
          ) : (
            <Stage
              ref={stageRef}
              width={canvasSize.width}
              height={canvasSize.height}
              onMouseDown={handleMouseDown}
              onMouseMove={handleMouseMove}
              onMouseUp={handleMouseUp}
              onMouseLeave={handleMouseUp}
            >
              <Layer>
                {/* Image */}
                {image && (
                  <KonvaImage
                    image={image}
                    x={imageScale.offsetX}
                    y={imageScale.offsetY}
                    width={currentImage.original_width * imageScale.scale}
                    height={currentImage.original_height * imageScale.scale}
                  />
                )}

                {/* Grid overlay */}
                {showGrid && image && currentImage && (() => {
                  const imgW = currentImage.original_width * imageScale.scale
                  const imgH = currentImage.original_height * imageScale.scale
                  const gridLines = []
                  const gridSize = 10 // 10x10 grid

                  // Vertical lines
                  for (let i = 1; i < gridSize; i++) {
                    const x = imageScale.offsetX + (imgW / gridSize) * i
                    gridLines.push(
                      <Line
                        key={`v${i}`}
                        points={[x, imageScale.offsetY, x, imageScale.offsetY + imgH]}
                        stroke="rgba(255, 255, 255, 0.4)"
                        strokeWidth={1}
                      />
                    )
                  }

                  // Horizontal lines
                  for (let i = 1; i < gridSize; i++) {
                    const y = imageScale.offsetY + (imgH / gridSize) * i
                    gridLines.push(
                      <Line
                        key={`h${i}`}
                        points={[imageScale.offsetX, y, imageScale.offsetX + imgW, y]}
                        stroke="rgba(255, 255, 255, 0.4)"
                        strokeWidth={1}
                      />
                    )
                  }

                  return gridLines
                })()}

                {/* Existing annotations */}
                {annotations.map((ann, idx) => {
                  const rect = yoloToCanvas(ann)
                  const cls = classes.find(c => c.class_index === ann.class_id)
                  const color = cls?.color || CLASS_COLORS[ann.class_id % CLASS_COLORS.length]
                  const isSelected = selectedAnnotations.has(idx)

                  return (
                    <Rect
                      key={idx}
                      {...rect}
                      stroke={isSelected ? '#FFFFFF' : color}
                      strokeWidth={isSelected ? 3 : 2}
                      fill={isSelected ? `${color}55` : 'transparent'}
                      dash={isSelected ? [8, 4] : undefined}
                      onClick={() => handleAnnotationClick(idx)}
                      onTap={() => handleAnnotationClick(idx)}
                    />
                  )
                })}

                {/* Drawing new rect */}
                {newRect && (
                  <Rect
                    {...newRect}
                    stroke={classes[selectedClassIdx]?.color || CLASS_COLORS[selectedClassIdx % CLASS_COLORS.length]}
                    strokeWidth={2}
                    dash={[5, 5]}
                    fill="transparent"
                  />
                )}
              </Layer>
            </Stage>
          )}
        </div>

        {/* Right sidebar - Annotations & Navigation */}
        <div className="w-56 bg-gray-50 border-l flex flex-col">
          {/* Navigation */}
          <div className="p-3 border-b">
            <div className="flex items-center justify-between mb-2">
              <button
                onClick={goToPrevious}
                disabled={absoluteIndex === 0}
                className="p-2 hover:bg-gray-200 rounded disabled:opacity-50"
              >
                <ArrowLeft size={20} />
              </button>
              <span className="text-sm font-medium">
                {(absoluteIndex + 1).toLocaleString()} / {totalImages.toLocaleString()}
              </span>
              <button
                onClick={goToNext}
                disabled={absoluteIndex >= totalImages - 1}
                className="p-2 hover:bg-gray-200 rounded disabled:opacity-50"
              >
                <ArrowRight size={20} />
              </button>
            </div>

            {/* Progress */}
            <div className="w-full bg-gray-200 rounded-full h-1.5">
              <div
                className="bg-emerald-500 h-1.5 rounded-full"
                style={{ width: `${((absoluteIndex + 1) / totalImages) * 100}%` }}
              />
            </div>

            {/* Delete image button */}
            <button
              onClick={handleDeleteImage}
              disabled={images.length === 0}
              className="w-full mt-2 px-3 py-1.5 text-sm bg-red-50 text-red-600 rounded hover:bg-red-100 disabled:opacity-50 flex items-center justify-center gap-1"
              title="Delete current image (Del)"
            >
              <Trash2 size={14} />
              Delete Image
            </button>
          </div>

          {/* Annotations list */}
          <div className="flex-1 overflow-y-auto p-2">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium">
                Annotations ({annotations.length})
                {selectedAnnotations.size > 0 && (
                  <span className="ml-1 text-emerald-600">
                    - {selectedAnnotations.size} selected
                  </span>
                )}
              </span>
              {annotations.length > 0 && (
                <button
                  onClick={() => {
                    if (confirm('Clear all annotations?')) {
                      setAnnotations([])
                      saveAnnotations([])
                      setSelectedAnnotations(new Set())
                    }
                  }}
                  className="text-red-500 hover:text-red-700"
                  title="Clear all"
                >
                  <Trash2 size={14} />
                </button>
              )}
            </div>

            {/* Bulk actions bar - shown when annotations are selected */}
            {selectedAnnotations.size > 0 && (
              <div className="mb-2 p-2 bg-emerald-50 rounded-lg border border-emerald-200">
                <div className="text-xs text-emerald-700 font-medium mb-2">
                  Bulk Actions ({selectedAnnotations.size} selected)
                </div>
                <div className="flex flex-col gap-1">
                  {/* Change class dropdown */}
                  <select
                    onChange={(e) => {
                      if (e.target.value !== '') {
                        changeSelectedAnnotationsClass(parseInt(e.target.value))
                        e.target.value = ''
                      }
                    }}
                    className="w-full text-xs border border-emerald-300 rounded px-2 py-1.5 bg-white"
                    defaultValue=""
                  >
                    <option value="" disabled>Change class to...</option>
                    {classes.map((cls, idx) => (
                      <option key={cls.id} value={cls.class_index}>
                        {idx + 1}. {cls.name}
                      </option>
                    ))}
                  </select>
                  <div className="flex gap-1">
                    <button
                      onClick={() => setSelectedAnnotations(new Set())}
                      className="flex-1 text-xs px-2 py-1.5 bg-gray-100 text-gray-600 rounded hover:bg-gray-200"
                    >
                      Deselect
                    </button>
                    <button
                      onClick={deleteSelectedAnnotations}
                      className="flex-1 text-xs px-2 py-1.5 bg-red-100 text-red-600 rounded hover:bg-red-200 flex items-center justify-center gap-1"
                    >
                      <Trash2 size={12} />
                      Delete
                    </button>
                  </div>
                </div>
                <div className="text-xs text-emerald-600 mt-1.5">
                  Tip: Press 1-9 to change class, Del to delete
                </div>
              </div>
            )}

            {annotations.length === 0 ? (
              <p className="text-sm text-gray-500 text-center py-4">
                Draw a box to annotate
              </p>
            ) : (
              <div className="space-y-1">
                {annotations.map((ann, idx) => {
                  const cls = classes.find(c => c.class_index === ann.class_id)
                  const isSelected = selectedAnnotations.has(idx)
                  return (
                    <div
                      key={idx}
                      onClick={() => handleAnnotationClick(idx)}
                      className={`flex items-center gap-2 p-2 rounded cursor-pointer ${
                        isSelected ? 'bg-emerald-100 ring-2 ring-emerald-400' : 'hover:bg-gray-100'
                      }`}
                    >
                      <input
                        type="checkbox"
                        checked={isSelected}
                        onChange={() => handleAnnotationClick(idx)}
                        onClick={(e) => e.stopPropagation()}
                        className="w-3.5 h-3.5 text-emerald-600 rounded border-gray-300 focus:ring-emerald-500"
                      />
                      <div
                        className="w-3 h-3 rounded"
                        style={{ backgroundColor: cls?.color || CLASS_COLORS[ann.class_id % CLASS_COLORS.length] }}
                      />
                      <span className="flex-1 text-sm truncate">{cls?.name || `Class ${ann.class_id}`}</span>
                      <button
                        onClick={(e) => {
                          e.stopPropagation()
                          deleteAnnotation(idx)
                        }}
                        className="text-gray-400 hover:text-red-500"
                      >
                        <X size={14} />
                      </button>
                    </div>
                  )
                })}
              </div>
            )}
          </div>

          {/* Image thumbnails */}
          <div className="p-2 border-t max-h-32 overflow-y-auto">
            <div className="flex flex-wrap gap-1">
              {images.slice(Math.max(0, currentIndex - 5), currentIndex + 10).map((img, idx) => {
                const actualIdx = Math.max(0, currentIndex - 5) + idx
                return (
                  <button
                    key={img.id}
                    onClick={() => setCurrentIndex(actualIdx)}
                    className={`w-8 h-8 rounded border-2 flex items-center justify-center text-xs ${
                      actualIdx === currentIndex
                        ? 'border-emerald-500 bg-emerald-100'
                        : img.is_annotated
                        ? 'border-green-300 bg-green-50'
                        : 'border-gray-200'
                    }`}
                  >
                    {img.is_annotated && <Check size={12} className="text-green-600" />}
                  </button>
                )
              })}
            </div>
          </div>
        </div>
      </div>

      {/* Upload Modal */}
      {showUploadModal && (
        <div className="fixed inset-0 bg-gray-600 bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-lg p-6 max-w-md w-full">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-xl font-bold">Upload Images</h2>
              <button onClick={() => setShowUploadModal(false)} className="text-gray-500" disabled={uploading}>
                <X size={24} />
              </button>
            </div>

            {uploading ? (
              <div className="py-8 text-center">
                <p className="mb-2 font-medium">Uploading... {uploadProgress}%</p>
                <div className="w-full bg-gray-200 rounded-full h-2 mb-2">
                  <div
                    className="bg-emerald-500 h-2 rounded-full transition-all"
                    style={{ width: `${uploadProgress}%` }}
                  />
                </div>
                <p className="text-sm text-gray-500">Please wait...</p>
              </div>
            ) : (
              <>
                {/* Drag and drop area */}
                <div
                  {...getRootProps()}
                  className={`border-2 border-dashed rounded-lg p-6 text-center cursor-pointer mb-4 ${
                    isDragActive ? 'border-emerald-500 bg-emerald-50' : 'border-gray-300'
                  }`}
                >
                  <input {...getInputProps()} />
                  <Upload size={32} className="mx-auto mb-2 text-gray-400" />
                  <p className="text-gray-600 font-medium">Drag and drop images</p>
                  <p className="text-sm text-gray-500">or click to select files</p>
                </div>

                {/* Folder upload option */}
                <div className="text-center">
                  <p className="text-sm text-gray-500 mb-2">Or upload an entire folder:</p>
                  <label className="inline-flex items-center gap-2 px-4 py-2 bg-blue-100 text-blue-700 rounded-lg cursor-pointer hover:bg-blue-200">
                    <FolderOpen size={18} />
                    Select Folder
                    <input
                      type="file"
                      webkitdirectory=""
                      directory=""
                      multiple
                      className="hidden"
                      accept="image/*"
                      onChange={(e) => {
                        const files = Array.from(e.target.files || []).filter(f =>
                          f.type.startsWith('image/') ||
                          /\.(jpg|jpeg|png|bmp|gif)$/i.test(f.name)
                        )
                        if (files.length > 0) {
                          onDrop(files)
                        } else {
                          alert('No image files found in the selected folder')
                        }
                        e.target.value = '' // Reset for re-selection
                      }}
                    />
                  </label>
                </div>

                <p className="text-xs text-gray-400 mt-4 text-center">
                  Supported formats: JPG, PNG, BMP, GIF
                </p>
              </>
            )}
          </div>
        </div>
      )}

      {/* Add Class Modal */}
      {showAddClassModal && (
        <div className="fixed inset-0 bg-gray-600 bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-lg p-6 max-w-sm w-full">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-xl font-bold">Add Class</h2>
              <button onClick={() => setShowAddClassModal(false)} className="text-gray-500">
                <X size={24} />
              </button>
            </div>

            <input
              type="text"
              value={newClassName}
              onChange={(e) => setNewClassName(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleAddClass()}
              placeholder="Class name"
              className="w-full border border-gray-300 rounded-md p-2 mb-4"
              autoFocus
            />

            <div className="mb-4">
              <label className="block text-sm font-medium mb-2">Color</label>
              <div className="flex items-center gap-2">
                <input
                  type="color"
                  value={newClassColor}
                  onChange={(e) => setNewClassColor(e.target.value)}
                  className="w-10 h-10 rounded border cursor-pointer"
                />
                <div className="flex flex-wrap gap-1">
                  {CLASS_COLORS.map((color) => (
                    <button
                      key={color}
                      onClick={() => setNewClassColor(color)}
                      className={`w-6 h-6 rounded ${newClassColor === color ? 'ring-2 ring-offset-1 ring-gray-400' : ''}`}
                      style={{ backgroundColor: color }}
                    />
                  ))}
                </div>
              </div>
            </div>

            <div className="flex gap-2">
              <button
                onClick={() => setShowAddClassModal(false)}
                className="flex-1 px-4 py-2 border border-gray-300 rounded-lg"
              >
                Cancel
              </button>
              <button
                onClick={handleAddClass}
                className="flex-1 px-4 py-2 bg-emerald-600 text-white rounded-lg"
              >
                Add
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Manage Classes Modal */}
      {showManageClassesModal && (
        <div className="fixed inset-0 bg-gray-600 bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-lg p-6 max-w-lg w-full max-h-[80vh] overflow-hidden flex flex-col">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-xl font-bold">Manage Classes</h2>
              <button onClick={() => { setShowManageClassesModal(false); setEditingClass(null); }} className="text-gray-500">
                <X size={24} />
              </button>
            </div>

            {classes.length === 0 ? (
              <div className="text-center py-8 text-gray-500">
                <p>No classes defined yet.</p>
                <button
                  onClick={() => { setShowManageClassesModal(false); setShowAddClassModal(true); }}
                  className="mt-2 text-emerald-600 hover:underline"
                >
                  Add your first class
                </button>
              </div>
            ) : (
              <div className="flex-1 overflow-y-auto">
                <div className="space-y-2">
                  {classes.map((cls, idx) => (
                    <div key={cls.id} className="border rounded-lg p-3">
                      {editingClass?.id === cls.id ? (
                        <div className="space-y-3">
                          <input
                            type="text"
                            value={editClassName}
                            onChange={(e) => setEditClassName(e.target.value)}
                            className="w-full border rounded px-3 py-2"
                            autoFocus
                          />
                          <div className="flex items-center gap-2">
                            <label className="text-sm text-gray-600">Color:</label>
                            <input
                              type="color"
                              value={editClassColor}
                              onChange={(e) => setEditClassColor(e.target.value)}
                              className="w-8 h-8 rounded border cursor-pointer"
                            />
                            <div className="flex flex-wrap gap-1">
                              {CLASS_COLORS.map((color) => (
                                <button
                                  key={color}
                                  onClick={() => setEditClassColor(color)}
                                  className={`w-5 h-5 rounded ${editClassColor === color ? 'ring-2 ring-offset-1 ring-gray-400' : ''}`}
                                  style={{ backgroundColor: color }}
                                />
                              ))}
                            </div>
                          </div>
                          <div className="flex gap-2">
                            <button
                              onClick={() => { setEditingClass(null); setEditClassName(''); setEditClassColor(''); }}
                              className="flex-1 px-3 py-1.5 text-sm border rounded hover:bg-gray-50"
                            >
                              Cancel
                            </button>
                            <button
                              onClick={handleUpdateClass}
                              className="flex-1 px-3 py-1.5 text-sm bg-emerald-600 text-white rounded hover:bg-emerald-700"
                            >
                              Save
                            </button>
                          </div>
                        </div>
                      ) : (
                        <div className="flex items-center gap-3">
                          <div
                            className="w-6 h-6 rounded flex-shrink-0"
                            style={{ backgroundColor: cls.color || CLASS_COLORS[idx % CLASS_COLORS.length] }}
                          />
                          <div className="flex-1">
                            <div className="font-medium">{cls.name}</div>
                            <div className="text-xs text-gray-500">Index: {cls.class_index}</div>
                          </div>
                          <button
                            onClick={() => startEditingClass(cls)}
                            className="p-2 text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded"
                            title="Edit class"
                          >
                            <Edit3 size={16} />
                          </button>
                          <button
                            onClick={() => handleDeleteClass(cls.id)}
                            className="p-2 text-red-500 hover:text-red-700 hover:bg-red-50 rounded"
                            title="Delete class"
                          >
                            <Trash2 size={16} />
                          </button>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Orphaned Classes Section */}
            {annotationClassStats?.classes?.filter(c => c.is_orphaned).length > 0 && (
              <div className="mt-4 pt-4 border-t">
                <h4 className="font-medium text-orange-700 mb-2">Orphaned Classes (No Definition)</h4>
                <p className="text-xs text-gray-500 mb-2">
                  These class IDs exist in annotations but have no class definition.
                  You can delete all annotations with these IDs or remap them to existing classes.
                </p>
                <div className="space-y-2">
                  {annotationClassStats.classes.filter(c => c.is_orphaned).map((cls) => (
                    <div key={`orphan-${cls.class_id}`} className="flex items-center justify-between p-2 bg-orange-50 border border-orange-200 rounded">
                      <div>
                        <span className="font-medium text-orange-800">Unknown (ID: {cls.class_id})</span>
                        <span className="ml-2 text-sm text-orange-600">{cls.count} annotations</span>
                      </div>
                      <button
                        onClick={() => handleDeleteOrphanedClassAnnotations(cls.class_id, cls.count)}
                        className="p-1 text-red-500 hover:bg-red-100 rounded"
                        title="Delete all annotations with this class ID"
                      >
                        <Trash2 size={16} />
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            )}

            <div className="mt-4 pt-4 border-t space-y-2">
              <button
                onClick={() => { setShowManageClassesModal(false); handleOpenRemapModal(); }}
                disabled={remapLoading}
                className="w-full px-4 py-2 bg-blue-100 text-blue-700 rounded-lg hover:bg-blue-200 flex items-center justify-center gap-2 disabled:opacity-50"
              >
                {remapLoading ? (
                  <Loader2 size={18} className="animate-spin" />
                ) : (
                  <Shuffle size={18} />
                )}
                Remap Annotations to Different Classes
              </button>
              <div className="flex gap-2">
                <button
                  onClick={() => { setShowManageClassesModal(false); setShowAddClassModal(true); }}
                  className="flex-1 px-4 py-2 bg-emerald-100 text-emerald-700 rounded-lg hover:bg-emerald-200 flex items-center justify-center gap-2"
                >
                  <Plus size={18} />
                  Add New Class
                </button>
                <button
                  onClick={() => { setShowManageClassesModal(false); setEditingClass(null); }}
                  className="px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50"
                >
                  Done
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Split Modal */}
      {showSplitModal && (
        <div className="fixed inset-0 bg-gray-600 bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-lg p-6 max-w-md w-full">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-xl font-bold">Generate Splits</h2>
              <button onClick={() => setShowSplitModal(false)} className="text-gray-500">
                <X size={24} />
              </button>
            </div>

            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium mb-1">Train Ratio: {Math.round(splitConfig.train_ratio * 100)}%</label>
                <input
                  type="range"
                  min="0"
                  max="100"
                  value={splitConfig.train_ratio * 100}
                  onChange={(e) => setSplitConfig({ ...splitConfig, train_ratio: e.target.value / 100 })}
                  className="w-full"
                />
              </div>

              <div>
                <label className="block text-sm font-medium mb-1">Val Ratio: {Math.round(splitConfig.val_ratio * 100)}%</label>
                <input
                  type="range"
                  min="0"
                  max="100"
                  value={splitConfig.val_ratio * 100}
                  onChange={(e) => setSplitConfig({ ...splitConfig, val_ratio: e.target.value / 100 })}
                  className="w-full"
                />
              </div>

              <div>
                <label className="block text-sm font-medium mb-1">Test Ratio: {Math.round(splitConfig.test_ratio * 100)}%</label>
                <input
                  type="range"
                  min="0"
                  max="100"
                  value={splitConfig.test_ratio * 100}
                  onChange={(e) => setSplitConfig({ ...splitConfig, test_ratio: e.target.value / 100 })}
                  className="w-full"
                />
              </div>

              <p className="text-sm text-gray-500">
                Total: {Math.round((splitConfig.train_ratio + splitConfig.val_ratio + splitConfig.test_ratio) * 100)}%
                {Math.abs(splitConfig.train_ratio + splitConfig.val_ratio + splitConfig.test_ratio - 1) > 0.01 && (
                  <span className="text-red-500 ml-2">(Must equal 100%)</span>
                )}
              </p>
            </div>

            <div className="flex gap-2 mt-6">
              <button
                onClick={() => setShowSplitModal(false)}
                className="flex-1 px-4 py-2 border border-gray-300 rounded-lg"
              >
                Cancel
              </button>
              <button
                onClick={handleGenerateSplits}
                disabled={Math.abs(splitConfig.train_ratio + splitConfig.val_ratio + splitConfig.test_ratio - 1) > 0.01}
                className="flex-1 px-4 py-2 bg-purple-600 text-white rounded-lg disabled:opacity-50"
              >
                Generate
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Settings Modal - Preprocessing & Augmentation */}
      {showSettingsModal && (
        <div className="fixed inset-0 bg-gray-600 bg-opacity-50 flex items-center justify-center z-50 p-4 overflow-y-auto">
          <div className="bg-white rounded-lg p-6 max-w-2xl w-full max-h-[90vh] overflow-y-auto">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-xl font-bold">Project Settings</h2>
              <button onClick={() => setShowSettingsModal(false)} className="text-gray-500">
                <X size={24} />
              </button>
            </div>

            {/* Preprocessing Section */}
            <div className="mb-6">
              <h3 className="text-lg font-semibold mb-3 text-gray-800 border-b pb-2">Preprocessing (Applied at Export)</h3>

              <div className="space-y-4">
                {/* Resize */}
                <div className="flex items-center justify-between">
                  <div>
                    <label className="font-medium">Resize Images</label>
                    <p className="text-sm text-gray-500">Resize all images to a square target size</p>
                  </div>
                  <input
                    type="checkbox"
                    checked={preprocessConfig.resize_enabled}
                    onChange={(e) => setPreprocessConfig({...preprocessConfig, resize_enabled: e.target.checked})}
                    className="w-5 h-5"
                  />
                </div>

                {preprocessConfig.resize_enabled && (
                  <div className="ml-4 p-3 bg-gray-50 rounded-lg space-y-3">
                    <div className="flex items-center gap-4">
                      <label className="text-sm w-28">Target Size:</label>
                      <select
                        value={preprocessConfig.target_size}
                        onChange={(e) => setPreprocessConfig({...preprocessConfig, target_size: parseInt(e.target.value)})}
                        className="border rounded px-3 py-1"
                      >
                        <option value={480}>480x480 (fast training)</option>
                        <option value={640}>640x640 (recommended)</option>
                        <option value={960}>960x960 (high detail)</option>
                        <option value={1440}>1440x1440 (max quality)</option>
                      </select>
                    </div>

                    <div className="flex items-center gap-4">
                      <label className="text-sm w-28">Letterbox:</label>
                      <input
                        type="checkbox"
                        checked={preprocessConfig.letterbox}
                        onChange={(e) => setPreprocessConfig({...preprocessConfig, letterbox: e.target.checked})}
                      />
                      <span className="text-sm text-gray-500">Keep aspect ratio with padding</span>
                    </div>

                    {preprocessConfig.letterbox && (
                      <div className="flex items-center gap-4">
                        <label className="text-sm w-28">Pad Color:</label>
                        <input
                          type="color"
                          value={preprocessConfig.letterbox_color}
                          onChange={(e) => setPreprocessConfig({...preprocessConfig, letterbox_color: e.target.value})}
                          className="w-10 h-8 rounded border"
                        />
                        <span className="text-sm text-gray-500">{preprocessConfig.letterbox_color}</span>
                      </div>
                    )}
                  </div>
                )}

                {/* Auto Orient */}
                <div className="flex items-center justify-between">
                  <div>
                    <label className="font-medium">Auto-Orient (EXIF)</label>
                    <p className="text-sm text-gray-500">
                      Phones save rotation info in EXIF metadata instead of rotating the actual pixels.
                      This fixes images that appear sideways or upside-down.
                    </p>
                  </div>
                  <input
                    type="checkbox"
                    checked={preprocessConfig.auto_orient}
                    onChange={(e) => setPreprocessConfig({...preprocessConfig, auto_orient: e.target.checked})}
                    className="w-5 h-5"
                  />
                </div>

                {/* CLAHE */}
                <div className="flex items-center justify-between">
                  <div>
                    <label className="font-medium">CLAHE (Contrast Enhancement)</label>
                    <p className="text-sm text-gray-500">
                      Improves visibility in dark or low-contrast images. Good for night/indoor footage
                      where objects are hard to see.
                    </p>
                  </div>
                  <input
                    type="checkbox"
                    checked={preprocessConfig.clahe_enabled}
                    onChange={(e) => setPreprocessConfig({...preprocessConfig, clahe_enabled: e.target.checked})}
                    className="w-5 h-5"
                  />
                </div>

                {preprocessConfig.clahe_enabled && (
                  <div className="ml-4 p-3 bg-gray-50 rounded-lg">
                    <div className="flex items-center gap-4">
                      <label className="text-sm w-28">Strength:</label>
                      <input
                        type="range"
                        min="1"
                        max="10"
                        step="0.5"
                        value={preprocessConfig.clahe_clip_limit}
                        onChange={(e) => setPreprocessConfig({...preprocessConfig, clahe_clip_limit: parseFloat(e.target.value)})}
                        className="flex-1"
                      />
                      <span className="text-sm w-10">{preprocessConfig.clahe_clip_limit}</span>
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Augmentation Section */}
            <div className="mb-6">
              <h3 className="text-lg font-semibold mb-3 text-gray-800 border-b pb-2">Augmentation (Training Set Only)</h3>
              <p className="text-sm text-gray-500 mb-4">
                Creates variations of your images to help the model generalize better.
                Only applied to training images during export.
              </p>

              {/* Preset buttons */}
              <div className="flex gap-2 mb-4">
                <button
                  onClick={() => setAugmentConfig({
                    enabled: false, copies: 2, horizontal_flip: false, vertical_flip: false,
                    rotation_range: 0, brightness_range: 0, contrast_range: 0, saturation_range: 0,
                    noise_enabled: false, blur_enabled: false
                  })}
                  className={`px-3 py-2 rounded-lg text-sm font-medium ${!augmentConfig.enabled ? 'bg-gray-800 text-white' : 'bg-gray-100 hover:bg-gray-200'}`}
                >
                  None
                </button>
                <button
                  onClick={() => setAugmentConfig({
                    enabled: true, copies: 2, horizontal_flip: true, vertical_flip: false,
                    rotation_range: 10, brightness_range: 0.15, contrast_range: 0.15, saturation_range: 0.1,
                    noise_enabled: false, blur_enabled: false
                  })}
                  className={`px-3 py-2 rounded-lg text-sm font-medium ${augmentConfig.enabled && augmentConfig.copies === 2 && !augmentConfig.noise_enabled ? 'bg-emerald-600 text-white' : 'bg-emerald-100 text-emerald-700 hover:bg-emerald-200'}`}
                >
                  Light (2x)
                </button>
                <button
                  onClick={() => setAugmentConfig({
                    enabled: true, copies: 3, horizontal_flip: true, vertical_flip: false,
                    rotation_range: 15, brightness_range: 0.25, contrast_range: 0.2, saturation_range: 0.15,
                    noise_enabled: true, blur_enabled: false
                  })}
                  className={`px-3 py-2 rounded-lg text-sm font-medium ${augmentConfig.enabled && augmentConfig.copies === 3 ? 'bg-blue-600 text-white' : 'bg-blue-100 text-blue-700 hover:bg-blue-200'}`}
                >
                  Medium (3x)
                </button>
                <button
                  onClick={() => setAugmentConfig({
                    enabled: true, copies: 5, horizontal_flip: true, vertical_flip: true,
                    rotation_range: 25, brightness_range: 0.35, contrast_range: 0.3, saturation_range: 0.25,
                    noise_enabled: true, blur_enabled: true
                  })}
                  className={`px-3 py-2 rounded-lg text-sm font-medium ${augmentConfig.enabled && augmentConfig.copies === 5 ? 'bg-purple-600 text-white' : 'bg-purple-100 text-purple-700 hover:bg-purple-200'}`}
                >
                  Heavy (5x)
                </button>
              </div>

              {augmentConfig.enabled && (
                <div className="p-4 bg-gray-50 rounded-lg space-y-4">
                  {/* Output multiplier */}
                  <div className="flex items-center justify-between pb-3 border-b">
                    <span className="font-medium">Output multiplier</span>
                    <div className="flex items-center gap-2">
                      <button
                        onClick={() => setAugmentConfig({...augmentConfig, copies: Math.max(1, augmentConfig.copies - 1)})}
                        className="w-8 h-8 rounded bg-gray-200 hover:bg-gray-300 font-bold"
                      >-</button>
                      <span className="w-8 text-center font-bold">{augmentConfig.copies}x</span>
                      <button
                        onClick={() => setAugmentConfig({...augmentConfig, copies: Math.min(10, augmentConfig.copies + 1)})}
                        className="w-8 h-8 rounded bg-gray-200 hover:bg-gray-300 font-bold"
                      >+</button>
                    </div>
                  </div>

                  {/* Geometric transforms */}
                  <div>
                    <h4 className="text-sm font-semibold text-gray-700 mb-2">Geometric</h4>
                    <div className="grid grid-cols-2 gap-3">
                      <label className="flex items-center gap-2 p-2 rounded hover:bg-gray-100 cursor-pointer">
                        <input type="checkbox" checked={augmentConfig.horizontal_flip}
                          onChange={(e) => setAugmentConfig({...augmentConfig, horizontal_flip: e.target.checked})}
                          className="w-4 h-4" />
                        <span className="text-sm">Flip horizontal</span>
                      </label>
                      <label className="flex items-center gap-2 p-2 rounded hover:bg-gray-100 cursor-pointer">
                        <input type="checkbox" checked={augmentConfig.vertical_flip}
                          onChange={(e) => setAugmentConfig({...augmentConfig, vertical_flip: e.target.checked})}
                          className="w-4 h-4" />
                        <span className="text-sm">Flip vertical</span>
                      </label>
                    </div>
                    <div className="mt-2">
                      <div className="flex items-center justify-between text-sm mb-1">
                        <span>Rotation</span>
                        <span className="text-gray-500">±{augmentConfig.rotation_range}°</span>
                      </div>
                      <input type="range" min="0" max="45" value={augmentConfig.rotation_range}
                        onChange={(e) => setAugmentConfig({...augmentConfig, rotation_range: parseInt(e.target.value)})}
                        className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer" />
                    </div>
                  </div>

                  {/* Color transforms */}
                  <div>
                    <h4 className="text-sm font-semibold text-gray-700 mb-2">Color</h4>
                    <div className="space-y-2">
                      <div>
                        <div className="flex items-center justify-between text-sm mb-1">
                          <span>Brightness</span>
                          <span className="text-gray-500">±{Math.round(augmentConfig.brightness_range * 100)}%</span>
                        </div>
                        <input type="range" min="0" max="50" value={augmentConfig.brightness_range * 100}
                          onChange={(e) => setAugmentConfig({...augmentConfig, brightness_range: parseInt(e.target.value) / 100})}
                          className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer" />
                      </div>
                      <div>
                        <div className="flex items-center justify-between text-sm mb-1">
                          <span>Contrast</span>
                          <span className="text-gray-500">±{Math.round(augmentConfig.contrast_range * 100)}%</span>
                        </div>
                        <input type="range" min="0" max="50" value={augmentConfig.contrast_range * 100}
                          onChange={(e) => setAugmentConfig({...augmentConfig, contrast_range: parseInt(e.target.value) / 100})}
                          className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer" />
                      </div>
                      <div>
                        <div className="flex items-center justify-between text-sm mb-1">
                          <span>Saturation</span>
                          <span className="text-gray-500">±{Math.round(augmentConfig.saturation_range * 100)}%</span>
                        </div>
                        <input type="range" min="0" max="50" value={augmentConfig.saturation_range * 100}
                          onChange={(e) => setAugmentConfig({...augmentConfig, saturation_range: parseInt(e.target.value) / 100})}
                          className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer" />
                      </div>
                    </div>
                  </div>

                  {/* Noise/Blur */}
                  <div>
                    <h4 className="text-sm font-semibold text-gray-700 mb-2">Distortion</h4>
                    <div className="grid grid-cols-2 gap-3">
                      <label className="flex items-center gap-2 p-2 rounded hover:bg-gray-100 cursor-pointer">
                        <input type="checkbox" checked={augmentConfig.noise_enabled}
                          onChange={(e) => setAugmentConfig({...augmentConfig, noise_enabled: e.target.checked})}
                          className="w-4 h-4" />
                        <span className="text-sm">Gaussian noise</span>
                      </label>
                      <label className="flex items-center gap-2 p-2 rounded hover:bg-gray-100 cursor-pointer">
                        <input type="checkbox" checked={augmentConfig.blur_enabled}
                          onChange={(e) => setAugmentConfig({...augmentConfig, blur_enabled: e.target.checked})}
                          className="w-4 h-4" />
                        <span className="text-sm">Random blur</span>
                      </label>
                    </div>
                  </div>

                  {/* Preview Button */}
                  <div className="mt-4 pt-4 border-t">
                    <button
                      onClick={handleAugmentationPreview}
                      disabled={!currentImage || augmentPreviewLoading}
                      className="w-full px-4 py-2 bg-indigo-100 text-indigo-700 rounded-lg hover:bg-indigo-200 disabled:opacity-50 flex items-center justify-center gap-2"
                    >
                      {augmentPreviewLoading ? (
                        <>
                          <Loader2 size={16} className="animate-spin" />
                          Generating Preview...
                        </>
                      ) : (
                        <>
                          <Eye size={16} />
                          Preview Augmentation
                        </>
                      )}
                    </button>
                    <p className="text-xs text-gray-500 mt-1 text-center">
                      Generate sample augmented versions of the current image
                    </p>
                  </div>

                  {/* Preview Results */}
                  {augmentPreview && augmentPreview.previews && (
                    <div className="mt-4 p-4 bg-white border rounded-lg">
                      <div className="flex justify-between items-center mb-3">
                        <h4 className="font-medium text-gray-800">Augmentation Preview</h4>
                        <button
                          onClick={() => setAugmentPreview(null)}
                          className="text-gray-400 hover:text-gray-600"
                        >
                          <X size={16} />
                        </button>
                      </div>
                      <div className="grid grid-cols-2 gap-2">
                        {augmentPreview.previews.map((preview, idx) => (
                          <div key={idx} className="relative">
                            <img
                              src={`data:image/jpeg;base64,${preview.image}`}
                              alt={preview.type}
                              className="w-full rounded border"
                            />
                            <span className="absolute bottom-1 left-1 text-xs bg-black bg-opacity-50 text-white px-2 py-0.5 rounded">
                              {preview.type === 'original' ? 'Original' : `Augmented ${idx}`}
                            </span>
                          </div>
                        ))}
                      </div>
                      <p className="text-xs text-gray-500 mt-2 text-center">
                        {augmentPreview.previews.length} preview(s) - Bounding boxes are transformed with the image
                      </p>
                    </div>
                  )}
                </div>
              )}
            </div>

            <div className="flex gap-3 mt-6 pt-4 border-t">
              <button
                onClick={() => setShowSettingsModal(false)}
                className="flex-1 px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50"
              >
                Cancel
              </button>
              <button
                onClick={handleSaveSettings}
                className="flex-1 px-4 py-2 bg-emerald-600 text-white rounded-lg hover:bg-emerald-700"
              >
                Save Settings
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Remap Annotations Modal */}
      {showRemapModal && annotationClassStats && (
        <div className="fixed inset-0 bg-gray-600 bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-lg p-6 max-w-2xl w-full max-h-[80vh] overflow-hidden flex flex-col">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-xl font-bold">Remap Annotation Classes</h2>
              <button onClick={() => setShowRemapModal(false)} className="text-gray-500" disabled={remapLoading}>
                <X size={24} />
              </button>
            </div>

            <p className="text-sm text-gray-600 mb-4">
              Change the class assignment for all annotations. Select which class each current class should be remapped to.
              This is useful when you've imported data with different class IDs or want to merge classes.
            </p>

            {annotationClassStats.classes?.length === 0 ? (
              <div className="text-center py-8 text-gray-500">
                <p>No annotations found in this project.</p>
              </div>
            ) : (
              <div className="flex-1 overflow-y-auto">
                <table className="w-full">
                  <thead className="bg-gray-50 sticky top-0">
                    <tr>
                      <th className="text-left p-3 text-sm font-medium text-gray-600">Current Class</th>
                      <th className="text-center p-3 text-sm font-medium text-gray-600">Count</th>
                      <th className="text-center p-3 text-sm font-medium text-gray-600"></th>
                      <th className="text-left p-3 text-sm font-medium text-gray-600">Remap To</th>
                    </tr>
                  </thead>
                  <tbody>
                    {annotationClassStats.classes?.map((cls) => (
                      <tr key={cls.class_id} className={`border-b ${cls.is_orphaned ? 'bg-yellow-50' : ''}`}>
                        <td className="p-3">
                          <div className="flex items-center gap-2">
                            <div
                              className="w-4 h-4 rounded"
                              style={{ backgroundColor: cls.color }}
                            />
                            <span className={cls.is_orphaned ? 'text-yellow-700' : ''}>
                              {cls.name}
                            </span>
                            {cls.is_orphaned && (
                              <span className="text-xs bg-yellow-200 text-yellow-800 px-1.5 py-0.5 rounded">
                                Orphaned
                              </span>
                            )}
                          </div>
                          <div className="text-xs text-gray-400 mt-0.5">ID: {cls.class_id}</div>
                        </td>
                        <td className="p-3 text-center">
                          <span className="font-medium">{cls.count.toLocaleString()}</span>
                        </td>
                        <td className="p-3 text-center text-gray-400">
                          <ArrowRight size={16} />
                        </td>
                        <td className="p-3">
                          <select
                            value={remapMappings[cls.class_id] ?? cls.class_id}
                            onChange={(e) => handleRemapMappingChange(cls.class_id, e.target.value)}
                            className={`w-full border rounded px-3 py-2 ${
                              remapMappings[cls.class_id] !== cls.class_id
                                ? 'border-blue-500 bg-blue-50'
                                : 'border-gray-300'
                            }`}
                          >
                            {/* Option to keep as-is (same class_id) */}
                            <option value={cls.class_id}>
                              {cls.is_orphaned ? `Keep as ID ${cls.class_id}` : `${cls.name} (no change)`}
                            </option>
                            {/* Project classes */}
                            {annotationClassStats.project_classes?.filter(pc => pc.class_index !== cls.class_id).map((pc) => (
                              <option key={pc.class_index} value={pc.class_index}>
                                {pc.name} (ID: {pc.class_index})
                              </option>
                            ))}
                          </select>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}

            {/* Summary of changes */}
            {Object.entries(remapMappings).some(([old, newId]) => parseInt(old) !== parseInt(newId)) && (
              <div className="mt-4 p-3 bg-blue-50 rounded-lg">
                <h4 className="font-medium text-blue-800 mb-1">Changes to apply:</h4>
                <ul className="text-sm text-blue-700">
                  {Object.entries(remapMappings)
                    .filter(([old, newId]) => parseInt(old) !== parseInt(newId))
                    .map(([oldId, newId]) => {
                      const oldCls = annotationClassStats.classes?.find(c => c.class_id === parseInt(oldId))
                      const newCls = annotationClassStats.project_classes?.find(c => c.class_index === parseInt(newId))
                      return (
                        <li key={oldId}>
                          {oldCls?.name || `ID ${oldId}`} ({oldCls?.count || 0} annotations) → {newCls?.name || `ID ${newId}`}
                        </li>
                      )
                    })}
                </ul>
              </div>
            )}

            <div className="flex gap-3 mt-4 pt-4 border-t">
              <button
                onClick={() => setShowRemapModal(false)}
                disabled={remapLoading}
                className="flex-1 px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50 disabled:opacity-50"
              >
                Cancel
              </button>
              <button
                onClick={handleApplyRemap}
                disabled={remapLoading || !Object.entries(remapMappings).some(([old, newId]) => parseInt(old) !== parseInt(newId))}
                className="flex-1 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 flex items-center justify-center gap-2"
              >
                {remapLoading ? (
                  <>
                    <Loader2 size={18} className="animate-spin" />
                    Applying...
                  </>
                ) : (
                  <>
                    <Check size={18} />
                    Apply Remapping
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Quality Check Modal */}
      {showQualityModal && (
        <div className="fixed inset-0 bg-gray-600 bg-opacity-50 flex items-center justify-center z-50 p-4 overflow-y-auto">
          <div className="bg-white rounded-lg p-6 max-w-2xl w-full max-h-[90vh] overflow-y-auto">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-xl font-bold">Annotation Quality Check</h2>
              <button onClick={() => setShowQualityModal(false)} className="text-gray-500 hover:text-gray-700">
                <X size={24} />
              </button>
            </div>

            {qualityLoading ? (
              <div className="flex flex-col items-center justify-center py-12">
                <Loader2 size={48} className="animate-spin text-blue-600 mb-4" />
                <p className="text-gray-600">Analyzing annotations...</p>
              </div>
            ) : qualityResults ? (
              <div className="space-y-6">
                {/* Health Score */}
                <div className={`p-4 rounded-lg ${
                  qualityResults.health_score >= 90 ? 'bg-green-50 border border-green-200' :
                  qualityResults.health_score >= 70 ? 'bg-yellow-50 border border-yellow-200' :
                  'bg-red-50 border border-red-200'
                }`}>
                  <div className="flex items-center gap-3">
                    {qualityResults.health_score >= 90 ? (
                      <CheckCircle size={32} className="text-green-600" />
                    ) : qualityResults.health_score >= 70 ? (
                      <AlertTriangle size={32} className="text-yellow-600" />
                    ) : (
                      <AlertCircle size={32} className="text-red-600" />
                    )}
                    <div>
                      <p className="text-2xl font-bold">{qualityResults.health_score}%</p>
                      <p className="text-sm text-gray-600">
                        {qualityResults.health_score >= 90 ? 'Excellent annotation quality' :
                         qualityResults.health_score >= 70 ? 'Good quality with some issues' :
                         'Several quality issues detected'}
                      </p>
                    </div>
                  </div>
                </div>

                {/* Summary Stats */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                  <div className="bg-gray-50 p-3 rounded-lg text-center">
                    <p className="text-2xl font-bold text-gray-800">{qualityResults.total_images}</p>
                    <p className="text-xs text-gray-500">Total Images</p>
                  </div>
                  <div className="bg-gray-50 p-3 rounded-lg text-center">
                    <p className="text-2xl font-bold text-gray-800">{qualityResults.total_annotations}</p>
                    <p className="text-xs text-gray-500">Total Annotations</p>
                  </div>
                  <div className="bg-gray-50 p-3 rounded-lg text-center">
                    <p className="text-2xl font-bold text-gray-800">{qualityResults.images_with_annotations}</p>
                    <p className="text-xs text-gray-500">Annotated Images</p>
                  </div>
                  <div className="bg-gray-50 p-3 rounded-lg text-center">
                    <p className="text-2xl font-bold text-gray-800">
                      {qualityResults.total_images > 0
                        ? (qualityResults.total_annotations / qualityResults.images_with_annotations || 0).toFixed(1)
                        : 0}
                    </p>
                    <p className="text-xs text-gray-500">Avg. Per Image</p>
                  </div>
                </div>

                {/* Issues with Quick Fix Buttons */}
                <div>
                  <h3 className="font-semibold mb-3 text-gray-800">Issues Found</h3>
                  {qualityResults.issues && qualityResults.issues.length > 0 ? (
                    <div className="space-y-2 max-h-64 overflow-y-auto">
                      {qualityResults.issues.map((issue, idx) => (
                        <div
                          key={idx}
                          className={`flex items-start gap-2 p-3 rounded-lg ${
                            issue.severity === 'error' ? 'bg-red-50 text-red-800' :
                            issue.severity === 'warning' ? 'bg-yellow-50 text-yellow-800' :
                            'bg-blue-50 text-blue-800'
                          }`}
                        >
                          {issue.severity === 'error' ? (
                            <AlertCircle size={18} className="flex-shrink-0 mt-0.5" />
                          ) : (
                            <AlertTriangle size={18} className="flex-shrink-0 mt-0.5" />
                          )}
                          <div className="flex-1 min-w-0">
                            <p className="text-sm font-medium">{issue.type}</p>
                            <p className="text-xs opacity-75">{issue.message}</p>
                            {issue.count && (
                              <span className="inline-block mt-1 px-2 py-0.5 bg-white bg-opacity-50 rounded text-xs">
                                {issue.count} {issue.count === 1 ? 'instance' : 'instances'}
                              </span>
                            )}
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="text-center py-8 bg-green-50 rounded-lg">
                      <CheckCircle size={48} className="mx-auto text-green-600 mb-2" />
                      <p className="text-green-800 font-medium">No issues found!</p>
                      <p className="text-green-600 text-sm">Your annotations are in great shape.</p>
                    </div>
                  )}
                </div>

                {/* Quick Fix Tools */}
                {qualityResults.issues && qualityResults.issues.length > 0 && (
                  <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                    <h3 className="font-semibold text-blue-900 mb-3">Quick Fix Tools</h3>
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                      {qualityResults.summary?.small_boxes_count > 0 && (
                        <button
                          onClick={async () => {
                            if (!confirm(`Delete ${qualityResults.summary.small_boxes_count} small boxes (< 1% area)?`)) return
                            try {
                              const res = await annotationsAPI.fixDeleteSmallBoxes(projectId)
                              alert(`Deleted ${res.data.deleted} small boxes from ${res.data.affected_images} images`)
                              handleQualityCheck()
                            } catch (e) {
                              alert('Error: ' + (e.response?.data?.detail || e.message))
                            }
                          }}
                          className="flex items-center gap-2 px-3 py-2 bg-yellow-100 text-yellow-800 rounded-lg hover:bg-yellow-200 text-sm font-medium"
                        >
                          <Trash2 size={16} />
                          Delete Small Boxes ({qualityResults.summary.small_boxes_count})
                        </button>
                      )}
                      {qualityResults.summary?.large_boxes_count > 0 && (
                        <button
                          onClick={async () => {
                            if (!confirm(`Delete ${qualityResults.summary.large_boxes_count} large boxes (> 90% area)?`)) return
                            try {
                              const res = await annotationsAPI.fixDeleteLargeBoxes(projectId)
                              alert(`Deleted ${res.data.deleted} large boxes from ${res.data.affected_images} images`)
                              handleQualityCheck()
                            } catch (e) {
                              alert('Error: ' + (e.response?.data?.detail || e.message))
                            }
                          }}
                          className="flex items-center gap-2 px-3 py-2 bg-yellow-100 text-yellow-800 rounded-lg hover:bg-yellow-200 text-sm font-medium"
                        >
                          <Trash2 size={16} />
                          Delete Large Boxes ({qualityResults.summary.large_boxes_count})
                        </button>
                      )}
                      {qualityResults.summary?.overlapping_count > 0 && (
                        <button
                          onClick={async () => {
                            if (!confirm(`Merge ${qualityResults.summary.overlapping_count} overlapping box pairs (keep larger)?`)) return
                            try {
                              const res = await annotationsAPI.fixMergeOverlapping(projectId)
                              alert(`Removed ${res.data.deleted} duplicate boxes from ${res.data.affected_images} images`)
                              handleQualityCheck()
                            } catch (e) {
                              alert('Error: ' + (e.response?.data?.detail || e.message))
                            }
                          }}
                          className="flex items-center gap-2 px-3 py-2 bg-red-100 text-red-800 rounded-lg hover:bg-red-200 text-sm font-medium"
                        >
                          <Trash2 size={16} />
                          Merge Overlapping ({qualityResults.summary.overlapping_count})
                        </button>
                      )}
                      {qualityResults.summary?.unannotated_count > 0 && (
                        <button
                          onClick={async () => {
                            try {
                              const res = await annotationsAPI.getUnannotatedImages(projectId)
                              if (res.data.images.length > 0) {
                                const firstImage = res.data.images[0]
                                // Find index of this image in our images array
                                const idx = images.findIndex(img => img.id === firstImage.id)
                                if (idx !== -1) {
                                  setCurrentIndex(idx)
                                  setShowQualityModal(false)
                                } else {
                                  alert(`First unannotated image: ${firstImage.filename} (ID: ${firstImage.id})`)
                                }
                              }
                            } catch (e) {
                              alert('Error: ' + (e.response?.data?.detail || e.message))
                            }
                          }}
                          className="flex items-center gap-2 px-3 py-2 bg-blue-100 text-blue-800 rounded-lg hover:bg-blue-200 text-sm font-medium"
                        >
                          <ArrowRight size={16} />
                          Go to Unannotated ({qualityResults.summary.unannotated_count})
                        </button>
                      )}
                    </div>
                    <p className="text-xs text-blue-700 mt-2">
                      These tools automatically fix common annotation issues. Changes cannot be undone.
                    </p>
                  </div>
                )}

                {/* Tips */}
                <div className="text-sm text-gray-500 bg-gray-50 p-3 rounded-lg">
                  <p className="font-medium text-gray-700 mb-1">Quality Tips:</p>
                  <ul className="list-disc list-inside space-y-1">
                    <li>Boxes smaller than 1% of image area may be too small for detection</li>
                    <li>Boxes larger than 90% of image area may indicate labeling errors</li>
                    <li>Highly overlapping boxes (IoU &gt; 80%) may be duplicates</li>
                    <li>Edge-truncated boxes should match visible object boundaries</li>
                  </ul>
                </div>
              </div>
            ) : (
              <p className="text-center text-gray-500 py-8">No quality check results available.</p>
            )}

            <div className="flex gap-3 mt-6 pt-4 border-t">
              <button
                onClick={() => setShowQualityModal(false)}
                className="flex-1 px-4 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200"
              >
                Close
              </button>
              <button
                onClick={handleQualityCheck}
                disabled={qualityLoading}
                className="flex-1 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 flex items-center justify-center gap-2"
              >
                {qualityLoading ? (
                  <Loader2 size={18} className="animate-spin" />
                ) : (
                  <AlertTriangle size={18} />
                )}
                Re-check
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default AnnotateProject
