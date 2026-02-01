{{/*
Expand the name of the chart.
*/}}
{{- define "nmoe.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "nmoe.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "nmoe.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "nmoe.labels" -}}
helm.sh/chart: {{ include "nmoe.chart" . }}
{{ include "nmoe.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "nmoe.selectorLabels" -}}
app.kubernetes.io/name: {{ include "nmoe.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "nmoe.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "nmoe.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
SGLang fullname
*/}}
{{- define "nmoe.sglang.fullname" -}}
{{- printf "%s-sglang" (include "nmoe.fullname" .) | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
SkyRL fullname
*/}}
{{- define "nmoe.skyrl.fullname" -}}
{{- printf "%s-skyrl" (include "nmoe.fullname" .) | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
SGLang labels
*/}}
{{- define "nmoe.sglang.labels" -}}
{{ include "nmoe.labels" . }}
app.kubernetes.io/component: inference
{{- end }}

{{/*
SkyRL labels
*/}}
{{- define "nmoe.skyrl.labels" -}}
{{ include "nmoe.labels" . }}
app.kubernetes.io/component: training
{{- end }}

{{/*
SGLang selector labels
*/}}
{{- define "nmoe.sglang.selectorLabels" -}}
{{ include "nmoe.selectorLabels" . }}
app.kubernetes.io/component: inference
{{- end }}

{{/*
SkyRL selector labels
*/}}
{{- define "nmoe.skyrl.selectorLabels" -}}
{{ include "nmoe.selectorLabels" . }}
app.kubernetes.io/component: training
{{- end }}

{{/*
Return the proper image name
*/}}
{{- define "nmoe.image" -}}
{{- $registryName := .imageRoot.registry -}}
{{- $repositoryName := .imageRoot.repository -}}
{{- $tag := .imageRoot.tag | toString -}}
{{- if .global }}
    {{- if .global.imageRegistry }}
        {{- $registryName = .global.imageRegistry -}}
    {{- end -}}
{{- end -}}
{{- if $registryName }}
{{- printf "%s/%s:%s" $registryName $repositoryName $tag -}}
{{- else -}}
{{- printf "%s:%s" $repositoryName $tag -}}
{{- end -}}
{{- end -}}

{{/*
Generate CUDA environment variables
*/}}
{{- define "nmoe.cuda.env" -}}
- name: CUDA_VISIBLE_DEVICES
  value: "0,1,2,3,4,5,6,7"
- name: PYTORCH_CUDA_ALLOC_CONF
  value: "max_split_size_mb:{{ .Values.sglang.cuda.maxSplitSize | replace "m" "" }}"
- name: NCCL_DEBUG
  value: "INFO"
- name: NCCL_IB_DISABLE
  value: "0"
- name: NCCL_NET_GDR_LEVEL
  value: "5"
{{- end }}

{{/*
Generate RDEP environment variables
*/}}
{{- define "nmoe.rdep.env" -}}
- name: NMOE_RDEP_BACKEND
  value: {{ .Values.nmoe.rdep.backend | quote }}
- name: NMOE_RDEP_BUFFER_SIZE
  value: {{ .Values.nmoe.rdep.bufferSize | quote }}
- name: NMOE_RDEP_NUM_BUFFERS
  value: {{ .Values.nmoe.rdep.numBuffers | quote }}
- name: NMOE_RDEP_ASYNC
  value: {{ .Values.nmoe.rdep.enableAsyncDispatch | quote }}
{{- end }}
