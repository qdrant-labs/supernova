import {
  Component,
  ElementRef,
  HostListener,
  AfterViewInit,
  OnDestroy,
  OnInit,
  PLATFORM_ID,
  ViewChild,
  inject,
  signal,
  ChangeDetectionStrategy
} from '@angular/core';
import { DecimalPipe, isPlatformBrowser } from '@angular/common';
import { FormsModule } from '@angular/forms';
import * as d3 from 'd3';

import { Qdrant } from '../../qdrant/qdrant';
import {
  ForceGraphLink,
  ForceGraphNode,
  buildEmbeddingGraph,
} from './force-graph-builder';

interface ForceLayoutParams {
  chargeStrength: number;
  linkDistance: (link: ForceGraphLink) => number;
  collisionRadius: number;
  centerX: number;
  centerY: number;
  boundaryStrength: number;
  radialRadius: number;
  radialStrength: number;
}

interface ForceTuning {
  chargeMultiplier: number;
  linkDistanceMultiplier: number;
  linkStrengthMultiplier: number;
  collisionMultiplier: number;
  boundaryStrengthMultiplier: number;
  radialStrengthMultiplier: number;
  radialRadiusMultiplier: number;
}

@Component({
  selector: 'app-d3-visual',
  imports: [FormsModule, DecimalPipe],
  templateUrl: './d3-visual.html',
  changeDetection: ChangeDetectionStrategy.Eager,
  styleUrl: './d3-visual.css',
})
export class D3Visual implements OnInit, AfterViewInit, OnDestroy {
  private readonly qdrant = inject(Qdrant);
  private readonly platformId = inject(PLATFORM_ID);

  @ViewChild('graphPanel')
  private graphPanel?: ElementRef<HTMLDivElement>;

  @ViewChild('graphContainer', { static: true })
  private graphContainer!: ElementRef<HTMLDivElement>;

  protected readonly collections = signal<string[]>([]);
  protected readonly selectedCollection = signal('');
  protected readonly resultLimit = signal(40);
  protected readonly minSimilarity = signal(0.55);
  protected readonly status = signal('Select a collection and run a random dense vector query.');
  protected readonly error = signal('');
  protected readonly busy = signal(false);
  protected readonly nodeCount = signal(0);
  protected readonly linkCount = signal(0);
  protected readonly isMaximized = signal(false);

  protected readonly chargeMultiplier = signal(1);
  protected readonly linkDistanceMultiplier = signal(1);
  protected readonly linkStrengthMultiplier = signal(1);
  protected readonly collisionMultiplier = signal(1);
  protected readonly boundaryStrengthMultiplier = signal(1);
  protected readonly radialStrengthMultiplier = signal(1);
  protected readonly radialRadiusMultiplier = signal(1);
  protected readonly labelsOnHoverOnly = signal(false);

  private readonly graphPadding = 36;
  private graphWidth = 0;
  private graphHeight = 0;

  private resizeObserver?: ResizeObserver;
  private simulation?: d3.Simulation<ForceGraphNode, ForceGraphLink>;
  private svg?: d3.Selection<SVGSVGElement, unknown, null, undefined>;
  private graphGroup?: d3.Selection<SVGGElement, unknown, null, undefined>;
  private linkSelection?: d3.Selection<SVGLineElement, ForceGraphLink, SVGGElement, unknown>;
  private nodeSelection?: d3.Selection<SVGGElement, ForceGraphNode, SVGGElement, unknown>;

  ngOnInit(): void {
    if (isPlatformBrowser(this.platformId)) {
      void this.loadCollections();
      if (typeof ResizeObserver !== 'undefined') {
        this.resizeObserver = new ResizeObserver(() => {
          this.resizeGraph(true);
        });
        this.resizeObserver.observe(this.graphContainer.nativeElement);
      }
    } else {
      this.status.set('Open in the browser to render the force graph.');
    }
  }

  ngAfterViewInit(): void {
    if (isPlatformBrowser(this.platformId) && this.resizeObserver && this.graphPanel) {
      this.resizeObserver.observe(this.graphPanel.nativeElement);
    }
  }

  ngOnDestroy(): void {
    this.resizeObserver?.disconnect();
    this.simulation?.stop();
    this.setBodyScrollLocked(false);
  }

  @HostListener('document:keydown.escape')
  onEscapeKey(): void {
    if (this.isMaximized()) {
      this.toggleMaximize();
    }
  }

  toggleMaximize(): void {
    this.isMaximized.update((value) => !value);
    this.setBodyScrollLocked(this.isMaximized());

    if (isPlatformBrowser(this.platformId)) {
      this.scheduleGraphRelayout();
    }
  }

  @HostListener('window:resize')
  onWindowResize(): void {
    this.scheduleGraphRelayout();
  }

  private scheduleGraphRelayout(): void {
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        this.resizeGraph(true);
        requestAnimationFrame(() => this.resizeGraph(true));
      });
    });

    setTimeout(() => this.resizeGraph(true), 50);
  }

  async loadCollections(): Promise<void> {
    try {
      const names = await this.qdrant.getCollections();
      this.collections.set(names);
      if (names.length > 0 && !this.selectedCollection()) {
        this.selectedCollection.set(names[0]);
      }
      this.status.set(
        names.length > 0
          ? 'Ready to query with a random dense vector.'
          : 'No collections found.',
      );
    } catch {
      this.error.set('Failed to load collections from Qdrant via backend API.');
    }
  }

  protected onLabelsOnHoverOnlyChange(): void {
    this.applyLabelVisibility();
  }

  protected onForceParamChange(): void {
    if (!this.simulation || this.graphWidth === 0) {
      return;
    }

    this.applyForceLayout(
      this.graphWidth,
      this.graphHeight,
      this.simulation.nodes().length,
    );
    this.simulation.alpha(0.5).restart();
  }

  protected resetForceParams(): void {
    this.chargeMultiplier.set(1);
    this.linkDistanceMultiplier.set(1);
    this.linkStrengthMultiplier.set(1);
    this.collisionMultiplier.set(1);
    this.boundaryStrengthMultiplier.set(1);
    this.radialStrengthMultiplier.set(1);
    this.radialRadiusMultiplier.set(1);
    this.onForceParamChange();
  }

  private getForceTuning(): ForceTuning {
    return {
      chargeMultiplier: this.chargeMultiplier(),
      linkDistanceMultiplier: this.linkDistanceMultiplier(),
      linkStrengthMultiplier: this.linkStrengthMultiplier(),
      collisionMultiplier: this.collisionMultiplier(),
      boundaryStrengthMultiplier: this.boundaryStrengthMultiplier(),
      radialStrengthMultiplier: this.radialStrengthMultiplier(),
      radialRadiusMultiplier: this.radialRadiusMultiplier(),
    };
  }

  private applyTuning(layout: ForceLayoutParams, tuning: ForceTuning): ForceLayoutParams {
    const linkDistance = layout.linkDistance;

    return {
      ...layout,
      chargeStrength: layout.chargeStrength * tuning.chargeMultiplier,
      linkDistance: (link) => linkDistance(link) * tuning.linkDistanceMultiplier,
      collisionRadius: layout.collisionRadius * tuning.collisionMultiplier,
      boundaryStrength: layout.boundaryStrength * tuning.boundaryStrengthMultiplier,
      radialStrength: layout.radialStrength * tuning.radialStrengthMultiplier,
      radialRadius: layout.radialRadius * tuning.radialRadiusMultiplier,
    };
  }

  async runQuery(): Promise<void> {
    const collectionName = this.selectedCollection().trim();
    if (!collectionName || this.busy()) {
      return;
    }

    this.busy.set(true);
    this.error.set('');
    this.status.set('Running random dense vector query...');

    try {
      const result = await this.qdrant.randomDenseQuery(collectionName, this.resultLimit());
      if (result.points.length === 0) {
        throw new Error('No embeddings returned from the query.');
      }

      const graph = buildEmbeddingGraph(result.points, {
        minSimilarity: this.minSimilarity(),
        queryVector: result.queryVector,
      });

      const resultCount = result.points.length;
      this.nodeCount.set(graph.nodes.length);
      this.linkCount.set(graph.links.length);
      this.status.set(
        `Built force graph: 1 query node and ${resultCount} result${resultCount === 1 ? '' : 's'} (random ${result.vectorName ?? 'default'} vector).`,
      );
      this.renderGraph(graph.nodes, graph.links);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Force graph query failed.';
      this.error.set(message);
      this.status.set('Query failed.');
      this.clearGraph();
    } finally {
      this.busy.set(false);
    }
  }

  private resizeGraph(force = false): void {
    if (!this.svg || !this.simulation) {
      return;
    }

    const oldWidth = this.graphWidth;
    const oldHeight = this.graphHeight;
    const { width, height } = this.updateGraphDimensions();

    if (height < 100 && this.isMaximized()) {
      this.scheduleGraphRelayout();
      return;
    }

    if (!force && width === oldWidth && height === oldHeight) {
      return;
    }

    this.svg.attr('width', width).attr('height', height).attr('viewBox', `0 0 ${width} ${height}`);

    if (oldWidth > 0 && oldHeight > 0 && (width !== oldWidth || height !== oldHeight)) {
      this.rescaleNodes(oldWidth, oldHeight, width, height);
    }

    this.updateSimulationForDimensions(width, height, oldWidth, oldHeight);

    for (const node of this.simulation.nodes()) {
      this.clampNode(node, width, height);
    }

    this.updateGraphPositions();
    this.simulation.alpha(0.15).restart();
  }

  private computeForceLayoutParams(
    width: number,
    height: number,
    _nodeCount: number,
  ): ForceLayoutParams {
    return {
      chargeStrength: -120,
      linkDistance: (link) => 90 * (1 - link.similarity + 0.1),
      collisionRadius: 18,
      centerX: width / 2,
      centerY: height / 2,
      boundaryStrength: 0,
      radialRadius: 0,
      radialStrength: 0,
    };
  }

  private applyForceLayout(width: number, height: number, nodeCount: number): void {
    if (!this.simulation) {
      return;
    }

    const base = this.computeForceLayoutParams(width, height, nodeCount);
    const layout = this.applyTuning(base, this.getForceTuning());
    const tuning = this.getForceTuning();

    this.simulation.force('center', d3.forceCenter(layout.centerX, layout.centerY));

    if (layout.boundaryStrength > 0) {
      this.simulation
        .force('x', d3.forceX(layout.centerX).strength(layout.boundaryStrength))
        .force('y', d3.forceY(layout.centerY).strength(layout.boundaryStrength));
    } else {
      this.simulation.force('x', null).force('y', null);
    }

    if (layout.radialStrength > 0 && layout.radialRadius > 0) {
      this.simulation.force(
        'radial',
        d3
          .forceRadial(layout.radialRadius, layout.centerX, layout.centerY)
          .strength(layout.radialStrength),
      );
    } else {
      this.simulation.force('radial', null);
    }

    this.simulation
      .force('charge', d3.forceManyBody().strength(layout.chargeStrength))
      .force('collision', d3.forceCollide<ForceGraphNode>().radius(layout.collisionRadius));

    const linkForce = this.simulation.force('link') as d3.ForceLink<
      ForceGraphNode,
      ForceGraphLink
    > | null;

    if (linkForce) {
      linkForce
        .distance(layout.linkDistance)
        .strength((link) => link.similarity * tuning.linkStrengthMultiplier);
    }
  }

  private updateSimulationForDimensions(
    width: number,
    height: number,
    _oldWidth: number,
    _oldHeight: number,
  ): void {
    if (!this.simulation) {
      return;
    }

    this.applyForceLayout(width, height, this.simulation.nodes().length);
  }

  private rescaleNodes(
    oldWidth: number,
    oldHeight: number,
    newWidth: number,
    newHeight: number,
  ): void {
    if (!this.simulation) {
      return;
    }

    const scaleX = newWidth / oldWidth;
    const scaleY = newHeight / oldHeight;
    const oldCenterX = oldWidth / 2;
    const oldCenterY = oldHeight / 2;
    const newCenterX = newWidth / 2;
    const newCenterY = newHeight / 2;

    for (const node of this.simulation.nodes()) {
      if (node.x !== undefined && node.y !== undefined) {
        node.x = newCenterX + (node.x - oldCenterX) * scaleX;
        node.y = newCenterY + (node.y - oldCenterY) * scaleY;
      }

      if (node.fx != null) {
        node.fx = newCenterX + (node.fx - oldCenterX) * scaleX;
      }
      if (node.fy != null) {
        node.fy = newCenterY + (node.fy - oldCenterY) * scaleY;
      }
    }
  }

  private updateGraphDimensions(): { width: number; height: number } {
    const container = this.graphContainer.nativeElement;
    const containerRect = container.getBoundingClientRect();
    let width = Math.floor(containerRect.width);
    let height = Math.floor(containerRect.height);

    if (this.isMaximized()) {
      const panelRect = this.graphPanel?.nativeElement.getBoundingClientRect();
      const toolbar = this.graphPanel?.nativeElement.querySelector('.graph-panel__toolbar');
      const toolbarHeight = toolbar?.getBoundingClientRect().height ?? 0;
      const panelPadding = 32;

      if (panelRect) {
        width = Math.max(width, Math.floor(panelRect.width));
        height = Math.max(
          height,
          Math.floor(panelRect.height - toolbarHeight - panelPadding),
        );
      }

      if (height < 200) {
        width = Math.max(width, window.innerWidth - panelPadding);
        height = Math.max(height, window.innerHeight - toolbarHeight - panelPadding - 16);
      }
    } else {
      width = Math.max(width, 320);
      height = Math.max(height, 320);
    }

    this.graphWidth = width;
    this.graphHeight = height;
    return { width, height };
  }

  private getBounds(width = this.graphWidth, height = this.graphHeight) {
    const inset = this.graphPadding;
    return {
      minX: inset,
      maxX: width - inset,
      minY: inset,
      maxY: height - inset,
    };
  }

  private clampNode(node: ForceGraphNode, width: number, height: number): void {
    const { minX, maxX, minY, maxY } = this.getBounds(width, height);
    node.x = Math.max(minX, Math.min(maxX, node.x ?? width / 2));
    node.y = Math.max(minY, Math.min(maxY, node.y ?? height / 2));

    if (node.fx !== null && node.fx !== undefined) {
      node.fx = Math.max(minX, Math.min(maxX, node.fx));
    }
    if (node.fy !== null && node.fy !== undefined) {
      node.fy = Math.max(minY, Math.min(maxY, node.fy));
    }
  }

  private setBodyScrollLocked(locked: boolean): void {
    if (!isPlatformBrowser(this.platformId)) {
      return;
    }
    document.body.style.overflow = locked ? 'hidden' : '';
  }

  private clearGraph(): void {
    this.simulation?.stop();
    this.simulation = undefined;
    this.linkSelection = undefined;
    this.nodeSelection = undefined;
    this.nodeCount.set(0);
    this.linkCount.set(0);
    this.graphWidth = 0;
    this.graphHeight = 0;
    d3.select(this.graphContainer.nativeElement).select('svg').selectAll('*').remove();
    this.svg = undefined;
    this.graphGroup = undefined;
  }

  private renderGraph(nodes: ForceGraphNode[], links: ForceGraphLink[]): void {
    const container = this.graphContainer.nativeElement;
    const { width, height } = this.updateGraphDimensions();

    this.simulation?.stop();

    const svgSelection = d3.select(container).select<SVGSVGElement>('svg');
    svgSelection.selectAll('*').remove();

    this.svg = svgSelection
      .attr('width', width)
      .attr('height', height)
      .attr('viewBox', `0 0 ${width} ${height}`);
    this.graphGroup = this.svg.append('g');

    for (const node of nodes) {
      node.x = width / 2 + (Math.random() - 0.5) * width * 0.2;
      node.y = height / 2 + (Math.random() - 0.5) * height * 0.2;
      this.clampNode(node, width, height);
    }

    const nodeById = new Map(nodes.map((node) => [node.id, node]));
    const resolvedLinks = links.map((link) => ({
      ...link,
      source: nodeById.get(String(link.source)) ?? link.source,
      target: nodeById.get(String(link.target)) ?? link.target,
    }));

    this.linkSelection = this.graphGroup
      .append('g')
      .attr('class', 'links')
      .selectAll<SVGLineElement, ForceGraphLink>('line')
      .data(resolvedLinks)
      .join('line')
      .attr('class', 'link')
      .attr('stroke-width', (link) => 1 + link.similarity * 2);

    this.nodeSelection = this.graphGroup
      .append('g')
      .attr('class', 'nodes')
      .selectAll<SVGGElement, ForceGraphNode>('g')
      .data(nodes)
      .join('g')
      .attr('class', 'node');

    this.bindDragBehavior(this.nodeSelection);

    this.nodeSelection
      .append('circle')
      .attr('r', (node) => this.getNodeRadius(node))
      .attr('fill', (node) => this.getNodeFill(node));

    this.nodeSelection
      .append('text')
      .attr('class', 'node-label')
      .attr('x', 10)
      .attr('y', 4)
      .text((node) => node.label);

    this.bindLabelHoverBehavior(this.nodeSelection);
    this.applyLabelVisibility();

    this.simulation = d3.forceSimulation(nodes).force(
      'link',
      d3
        .forceLink<ForceGraphNode, ForceGraphLink>(resolvedLinks)
        .id((node) => node.id)
        .strength((link) => link.similarity),
    );

    this.applyForceLayout(width, height, nodes.length);
    this.simulation.on('tick', () => this.updateGraphPositions());
  }

  private updateGraphPositions(): void {
    if (!this.simulation || !this.linkSelection || !this.nodeSelection) {
      return;
    }

    const width = this.graphWidth;
    const height = this.graphHeight;

    for (const node of this.simulation.nodes()) {
      this.clampNode(node, width, height);
    }

    this.linkSelection
      .attr('x1', (link) => (link.source as ForceGraphNode).x ?? 0)
      .attr('y1', (link) => (link.source as ForceGraphNode).y ?? 0)
      .attr('x2', (link) => (link.target as ForceGraphNode).x ?? 0)
      .attr('y2', (link) => (link.target as ForceGraphNode).y ?? 0);

    this.nodeSelection.attr('transform', (node) => `translate(${node.x ?? 0},${node.y ?? 0})`);
  }

  private getNodeFill(node: ForceGraphNode): string {
    return node.isQuery ? '#dc3545' : '#0d6efd';
  }

  private getNodeRadius(node: ForceGraphNode): number {
    return node.isQuery ? 12 : node.score !== undefined ? 6 + node.score * 4 : 8;
  }

  private applyLabelVisibility(hoveredNode?: ForceGraphNode): void {
    if (!this.nodeSelection) {
      return;
    }

    const hoverOnly = this.labelsOnHoverOnly();

    this.nodeSelection.select<SVGTextElement>('.node-label').attr('visibility', (node) => {
      if (!hoverOnly) {
        return 'visible';
      }
      return node === hoveredNode ? 'visible' : 'hidden';
    });
  }

  private bindLabelHoverBehavior(
    selection: d3.Selection<SVGGElement, ForceGraphNode, SVGGElement, unknown>,
  ): void {
    selection
      .on('mouseenter', (_event, node) => {
        if (this.labelsOnHoverOnly()) {
          this.applyLabelVisibility(node);
        }
      })
      .on('mouseleave', () => {
        if (this.labelsOnHoverOnly()) {
          this.applyLabelVisibility();
        }
      });
  }

  private bindDragBehavior(
    selection: d3.Selection<SVGGElement, ForceGraphNode, SVGGElement, unknown>,
  ): void {
    const drag = d3
      .drag<SVGGElement, ForceGraphNode>()
      .on('start', (event, node) => {
        if (!event.active) {
          this.simulation?.alphaTarget(0.3).restart();
        }
        node.fx = node.x;
        node.fy = node.y;
      })
      .on('drag', (event, node) => {
        const { minX, maxX, minY, maxY } = this.getBounds();
        node.fx = Math.max(minX, Math.min(maxX, event.x));
        node.fy = Math.max(minY, Math.min(maxY, event.y));
      })
      .on('end', (event, node) => {
        if (!event.active) {
          this.simulation?.alphaTarget(0);
        }
        node.fx = null;
        node.fy = null;
      });

    selection.call(drag);
  }
}
