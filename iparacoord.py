import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from bokeh.io import output_notebook, push_notebook
from bokeh.plotting import figure, show
from bokeh.palettes import Category10, Inferno256
from bokeh.models import Label, Circle, Line, HoverTool
output_notebook()


cm = plt.get_cmap('winter')
colors = [cm(i) for i in range(cm.N)]
colors = ['#{:02x}{:02x}{:02x}'.format(int(255*r), int(255*g), int(255*b)) for r, g, b, _ in colors]


class ParallelCoordinatesPlot:
    pass


class ParallelCoordinatesTree:
    def __init__(self, estimator):
        if isinstance(estimator, tree.DecisionTreeClassifier):
            self.mode = 'classifier'
        elif isinstance(estimator, tree.DecisionTreeRegressor):
            self.mode = 'regressor'
        else:
            raise ValueError  # XXX born again trees
        self.estimator = estimator

    def recursive(self, i=0, depth=0, x_left=-1, x_right=1):
        df = self.df
        df.loc[i, 'depth'] = depth
        df.loc[i, 'x_left'] = x_left
        df.loc[i, 'x_right'] = x_right
        df.loc[i, 'x_center'] = (x_left + x_right) / 2
        feature = int(df.loc[i, 'feature'])
        if feature >= 0:
            left = df.loc[i, 'left']
            right = df.loc[i, 'right']
            threshold = df.loc[i, 'threshold']
            num_samples = df.loc[i, 'num_samples']
            # Compute coordinate
            x = self.X[self.decision_path[:, i] == 1, feature]
            v_min = x.min()
            v_max = x.max()
            ratio = (threshold - v_min) / (v_max - v_min)
            x_threshold = x_left + (x_right - x_left) * ratio
            df.loc[i, 'x_threshold'] = x_threshold
            df.loc[i, 'v_min'] = v_min
            df.loc[i, 'v_max'] = v_max
            # Compute for children
            ratio = df.loc[left, 'num_samples'] / num_samples
            x_split = x_left + (x_right - x_left) * ratio
            x_split_left = x_split - (x_split - x_left) * self.margin
            x_split_right = x_split + (x_right - x_split) * self.margin
            self.recursive(i=left, depth=depth+1, x_left=x_left, x_right=x_split_left)
            self.recursive(i=right, depth=depth+1, x_left=x_split_right, x_right=x_right)
    
    def get_color(self, y):
        if self.mode == 'classifier':
            return Category10[10][int(y)%10]
        elif self.mode == 'regressor':
            # return Inferno256[int((y - self.y_min) / (self.y_max - self.y_min) * 255)]
            return colors[int((y - self.y_min) / (self.y_max - self.y_min) * 255)]
    
    def get_scale_ratio(self, row, x):
        v_min = row['v_min']
        v_max = row['v_max']
        return (x - v_min) / (v_max - v_min)
    
    def get_scaled_x(self, row, x):
        ratio = self.get_scale_ratio(row, x)
        x_left = row['x_left']
        x_right = row['x_right']
        return x_left + (x_right - x_left) * ratio
    
    def visualize(
        self, X, y,
        feature_names=None, target_names=None,
        target_name=None, target_index=0,
        plot_width=800, plot_height=400,
        alpha=0.2,
        margin=0.05,
        tap=True, hover=True,
        hist=True, hist_bins=30,
    ):
        """
        Args:
            X (np.ndarray): feature
            y (np.ndarray): label
            
        """
        
        # Train tree if not trained yet
        if not hasattr(self.estimator, 'tree_'):
            self.estimator.fit(X, y)
        
        # Set default names if not specified
        if feature_names is None:
            feature_names = ['feature_'+str(i) for i in range(self.estimator.n_features_)]
        if target_names is None:
            if self.mode == 'classifier':
                target_names = ['class_'+str(i) for i in range(self.estimator.n_classes_)]
            elif self.mode == 'regressor':
                target_names = ['output_'+str(i) for i in range(self.estimator.n_outputs_)]
        self.feature_names = feature_names
        self.target_names = target_names
        
        # Set regression target
        if self.mode == 'regressor':
            if target_name is None:
                target_name = target_names[target_index]
            else:
                target_index = target_names.index(target_name)
        self.target_index = target_index
        
        # Get ground truth and make prediction
        y_true = y
        y_pred = self.estimator.predict(X)
        if self.estimator.n_outputs_ > 1:
            y_true = y_true[:, target_index]
            y_pred = y_pred[:, target_index]
        
        # Parse tree structure
        tree_ = self.estimator.tree_
        df = pd.DataFrame([], index=range(len(tree_.feature)))
        df['left'] = tree_.children_left
        df['right'] = tree_.children_right
        df['feature'] = tree_.feature
        df['threshold'] = tree_.threshold
        df['num_samples'] = tree_.n_node_samples
        self.df = df
        self.X = X
        self.margin = margin
        self.max_depth = tree_.max_depth
        self.decision_path = tree_.decision_path(X.astype(np.float32)).toarray()
        if self.mode == 'classifier':
            self.recursive()
        elif self.mode == 'regressor':
            self.y_min = y_true.min()
            self.y_max = y_true.max()
            self.recursive(x_left=self.y_min, x_right=self.y_max)
        
        # Create figure
        p = figure(
            plot_width=plot_width, plot_height=plot_height,
            tools='pan,xwheel_zoom,box_zoom,undo,reset,save,tap',
            active_scroll='xwheel_zoom',
        )
        
        # Plot paths
        lines = []
        for i, x in enumerate(X):
            xs = []
            ys = []
            nodes = np.where(self.decision_path[i] == 1)[0]
            for depth, node in enumerate(nodes):
                row = df.loc[node]
                feature = int(row['feature'])
                if feature >= 0:
                    xs.append(self.get_scaled_x(row, x[feature]))
                    ys.append(-depth)
                else:
                    if self.mode == 'classifier':
                        xs.append(row['x_center'])
                        ys.append(-self.max_depth)
                    elif self.mode == 'regressor':
                        xs.append(y_pred[i])
                        ys.append(-self.max_depth)
                        xs.append(y_true[i])
                        ys.append(-self.max_depth-1)
            color = self.get_color(y_true[i])
            line = p.line(xs, ys, alpha=alpha, color=color)
            if tap:
                line.selection_glyph = Line(line_alpha=1.0, line_color=color, line_width=2)
            if hover:
                line.hover_glyph = Line(line_alpha=1.0, line_color=color, line_width=2)
                lines.append(line)
        if hover:
            p.add_tools(HoverTool(tooltips=None, renderers=lines))
        
        # Plot histogram
        if hist:
            for index, row in self.df.iterrows():
                feature = int(row['feature'])
                if feature >= 0:
                    depth = row['depth']
                    cond = (self.decision_path[:, index] == 1)
                    x = X[cond, feature]
                    x = self.get_scaled_x(row, x)
                    hist_mean = len(x) / hist_bins * 10
                    if self.mode == 'classifier':
                        for i in range(self.estimator.n_classes_):
                            color = self.get_color(i)
                            x_class = x[y_true[cond] == i]
                            hist, edges = np.histogram(x_class, bins=hist_bins)
                            hist = hist / hist_mean - depth
                            p.quad(
                                top=hist, bottom=-depth, left=edges[:-1], right=edges[1:],
                                fill_color=color, line_color='white', alpha=0.5)
                    if self.mode == 'regressor':
                        hist, edges = np.histogram(x, bins=hist_bins)
                        hist = hist / hist_mean - depth
                        p.quad(
                            top=hist, bottom=-depth, left=edges[:-1], right=edges[1:],
                            fill_color='black', line_color='white', alpha=0.5)
        
        # Plot parallel coordinates
        for index, row in self.df.iterrows():
            feature = int(row['feature'])
            if feature >= 0:
                d = row['depth']
                xl = row['x_left']
                xr = row['x_right']
                xc = row['x_center']
                xt = row['x_threshold']
                threshold = row['threshold']
                name = feature_names[feature]
                p.line([xl, xr], [-d, -d], color='black')
                p.line([xl, xl], [-d+0.05, -d-0.05], color='black')
                p.line([xr, xr], [-d+0.05, -d-0.05], color='black')
                p.line([xt, xt], [-d, -d-0.05], color='black')
                p.add_layout(Label(
                    x=xc, y=-d+0.2, text='  {}  '.format(name),
                    text_align='center', text_baseline='bottom', text_font_size='10pt',
                    background_fill_color='white', background_fill_alpha=0.5))
                p.add_layout(Label(
                    x=xt, y=-d-0.1, text='{:.2f}'.format(threshold),
                    text_align='center', text_baseline='top', text_font_size='8pt',
                    background_fill_color='white', background_fill_alpha=0.5))
        
        # Adjust axis
        yticks = {-d: str(d+1) for d in range(0, self.max_depth)}
        yticks[-self.max_depth] = 'pred'
        if self.mode == 'classifier':
            for i, name in enumerate(target_names):
                p.line(0, 0, legend=name, color=self.get_color(i))
            p.legend.location = 'bottom_left'
            p.legend.background_fill_alpha = 0.5
            p.xaxis.visible = False
        elif self.mode == 'regressor':
            yticks[-self.max_depth-1] = 'true'
            p.xaxis.axis_label = target_names[target_index]
        p.xgrid.visible = False
        p.yaxis.axis_label = 'depth'
        p.yaxis.ticker = list(range(-self.max_depth-1, 1))
        p.yaxis.major_label_overrides = yticks
        p.yaxis.minor_tick_line_color = None
        
        # Show plot
        p.toolbar.logo = None
        self.t = show(p, notebook_handle=True)
        self.p = p
    
    def predict(self, x):
        y_pred = self.estimator.predict(x.reshape((1, -1)))
        if self.estimator.n_outputs_ > 1:
            y_pred = y_pred[:, self.target_index]
        
        # Display feature
        feature = pd.DataFrame([x], columns=self.feature_names)
        subset = feature.columns[self.estimator.tree_.feature]
        display(feature.style.background_gradient(cmap='gray', subset=subset))
        
        # Remove previous prediction path
        if hasattr(self, 'prediction_path'):
            self.p.renderers.remove(self.prediction_path)
        if hasattr(self, 'prediction_feature'):
            self.p.renderers.remove(self.prediction_feature)
        
        # Plot prediction path
        xs = []
        ys = []
        decision_path = self.estimator.tree_.decision_path(
            x.reshape((1, -1)).astype(np.float32)).toarray()
        nodes = np.where(decision_path == 1)[1]
        df = self.df
        for j, node in enumerate(nodes):
            feature = int(df.loc[node, 'feature'])
            if feature >= 0:
                v_min = df.loc[node, 'v_min']
                v_max = df.loc[node, 'v_max']
                x_left = df.loc[node, 'x_left']
                x_right = df.loc[node, 'x_right']
                ratio = (x[feature] - v_min) / (v_max - v_min)
                xs.append(x_left + (x_right - x_left) * ratio)
                ys.append(-j)
            else:
                if self.mode == 'classifier':
                    xs.append(df.loc[node, 'x_center'])
                    ys.append(-self.max_depth)
                elif self.mode == 'regressor':
                    xs.append(y_pred)
                    ys.append(-self.max_depth)
        self.prediction_path = self.p.line(xs, ys, color='black', alpha=0.5, line_width=3, line_cap='round')
        
        # Plot prediction feature
        xs = [xs[-1]]
        ys = [ys[-1]]
        for index, row in self.df.iterrows():
            feature = int(row['feature'])
            if feature >= 0:
                v_min = row['v_min']
                v_max = row['v_max']
                x_left = row['x_left']
                x_right = row['x_right']
                ratio = (x[feature] - v_min) / (v_max - v_min)
                if 0 <= ratio <= 1:
                    xs.append(x_left + (x_right - x_left) * ratio)
                    ys.append(-row['depth'])
        color = self.get_color(y_pred)
        self.prediction_path_circle = self.p.circle(xs, ys, color=color, alpha=0.5, size=10)
        
        # Update plot
        push_notebook(handle=self.t)
